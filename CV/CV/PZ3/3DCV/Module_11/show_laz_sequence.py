#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("TkAgg")          # works headless-free on Windows
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm


def intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Robust normalization: ignore extreme outliers and stretch useful range.
    p2, p98 = np.percentile(intensity, [2, 98])
    if p98 <= p2:
        p2 = np.min(intensity)
        p98 = np.max(intensity)
    span = max(p98 - p2, 1e-9)
    norm = np.clip((intensity - p2) / span, 0.0, 1.0)

    # Slight gamma correction makes mid-tones more visible.
    norm = np.power(norm, 0.8)

    # Blue -> Cyan -> Green -> Yellow -> Red
    anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    anchors_r = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    anchors_g = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    anchors_b = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    r = np.interp(norm, anchors_x, anchors_r)
    g = np.interp(norm, anchors_x, anchors_g)
    b = np.interp(norm, anchors_x, anchors_b)
    return np.column_stack((r, g, b))


def laz_to_pointcloud(file_path: Path, use_intensity_color: bool) -> o3d.geometry.PointCloud:
    las = laspy.read(str(file_path))
    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if use_intensity_color and hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)
        colors = intensity_to_rgb(intensity)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ---------------------------------------------------------------------------
# Spherical projection → regular point cloud
# ---------------------------------------------------------------------------

def make_regular_spherical_cloud(
    pcd: o3d.geometry.PointCloud,
    h_res_deg: float = 0.2,   # horizontal (azimuth) resolution in degrees
    v_res_deg: float = 0.2,   # vertical (elevation) resolution in degrees
) -> o3d.geometry.PointCloud:
    """
    Build a *regular* point cloud by projecting the input cloud onto a uniform
    spherical grid (range image) and mapping it back to 3-D Cartesian space.

    Steps
    -----
    1. Convert each point (x, y, z) → (r, azimuth φ, elevation θ).
    2. Bin azimuth and elevation into a uniform grid with the given resolutions.
    3. For each occupied cell keep the point with the *minimum* range (closest).
    4. Place the surviving points at the *centre* of their grid cell while keeping
       the measured range → regular angular sampling, real depth.

    The result looks like a LiDAR range-image unrolled back into 3-D.
    """
    xyz = np.asarray(pcd.points)
    if len(xyz) == 0:
        return o3d.geometry.PointCloud()

    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None

    # ---- 1. Cartesian → spherical ----------------------------------------
    r = np.linalg.norm(xyz, axis=1)                         # range
    valid = r > 1e-9
    safe_r = np.where(valid, r, 1.0)

    az = np.arctan2(xyz[:, 1], xyz[:, 0])                   # azimuth  [-π, π]
    el = np.where(
        valid,
        np.arcsin(np.clip(xyz[:, 2] / safe_r, -1.0, 1.0)),  # elevation [-π/2, π/2]
        0.0,
    )

    # ---- 2. Bin into uniform grid -----------------------------------------
    h_res = np.deg2rad(h_res_deg)
    v_res = np.deg2rad(v_res_deg)

    az_idx = np.floor((az + np.pi) / h_res).astype(np.int32)
    el_idx = np.floor((el + np.pi / 2.0) / v_res).astype(np.int32)

    # Encode 2-D cell address as a single integer key
    n_az = int(np.ceil(2.0 * np.pi / h_res)) + 2
    keys = el_idx * n_az + az_idx

    # ---- 3. Per-cell minimum-range selection (vectorised) -----------------
    # lexsort: primary key = cell key, secondary key = range (ascending)
    order = np.lexsort((r, keys))
    keys_sorted = keys[order]

    _, first_min_pos = np.unique(keys_sorted, return_index=True)
    selected = order[first_min_pos]          # indices into original arrays

    # ---- 4. Place at grid-cell centres with measured range ----------------
    az_reg = (az_idx[selected] + 0.5) * h_res - np.pi
    el_reg = (el_idx[selected] + 0.5) * v_res - np.pi / 2.0
    r_sel  = r[selected]

    cos_el = np.cos(el_reg)
    x_reg  = r_sel * cos_el * np.cos(az_reg)
    y_reg  = r_sel * cos_el * np.sin(az_reg)
    z_reg  = r_sel * np.sin(el_reg)

    reg_xyz = np.column_stack([x_reg, y_reg, z_reg])

    reg_pcd = o3d.geometry.PointCloud()
    reg_pcd.points = o3d.utility.Vector3dVector(reg_xyz)

    if has_colors and colors is not None and len(colors) > 0:
        reg_pcd.colors = o3d.utility.Vector3dVector(colors[selected])

    return reg_pcd


def make_range_image(
    pcd: o3d.geometry.PointCloud,
    h_res_deg: float = 0.2,
    v_res_deg: float = 0.2,
) -> np.ndarray:
    """
    Build a 2-D range image from the point cloud.
      rows    = elevation bins  (top = max elevation)
      columns = azimuth bins    (left = -180°, right = +180°)
      value   = range r  (0 means no measurement in that cell)
    """
    xyz = np.asarray(pcd.points)
    if len(xyz) == 0:
        return np.zeros((1, 1), dtype=np.float32)

    r = np.linalg.norm(xyz, axis=1)
    valid = r > 1e-9
    safe_r = np.where(valid, r, 1.0)

    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    el = np.where(valid, np.arcsin(np.clip(xyz[:, 2] / safe_r, -1.0, 1.0)), 0.0)

    h_res = np.deg2rad(h_res_deg)
    v_res = np.deg2rad(v_res_deg)

    n_az = int(np.ceil(2.0 * np.pi / h_res)) + 2
    n_el = int(np.ceil(np.pi / v_res)) + 2

    az_idx = np.clip(np.floor((az + np.pi) / h_res).astype(np.int32), 0, n_az - 1)
    el_idx = np.clip(np.floor((el + np.pi / 2.0) / v_res).astype(np.int32), 0, n_el - 1)

    keys = el_idx * n_az + az_idx
    order = np.lexsort((r, keys))
    _, first_min = np.unique(keys[order], return_index=True)
    selected = order[first_min]

    img = np.zeros((n_el, n_az), dtype=np.float32)
    img[el_idx[selected], az_idx[selected]] = r[selected].astype(np.float32)

    return np.flipud(img)   # elevation 0 at bottom


def save_range_image(
    pcd: o3d.geometry.PointCloud,
    out_path: Path,
    h_res_deg: float = 0.2,
    v_res_deg: float = 0.2,
) -> None:
    """Render the range image with plasma colormap and save as PNG."""
    img = make_range_image(pcd, h_res_deg=h_res_deg, v_res_deg=v_res_deg)

    fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
    filled = np.where(img > 0, img, np.nan)      # hide empty cells
    im = ax.imshow(
        filled,
        cmap="plasma",
        aspect="auto",
        interpolation="nearest",
        extent=[-180, 180, -90, 90],
    )
    plt.colorbar(im, ax=ax, label="Range (m)")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title("Spherical range image (regular grid)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Range image saved → {out_path}")


# ---------------------------------------------------------------------------

def collect_laz_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.laz"))
    if not files:
        raise FileNotFoundError(f"No .laz files found in: {folder}")
    return files


def convert_laz_folder_to_xyz(
    files: list[Path],
    output_dir: Path,
    xyz_fmt: str = "%.6f",
    delimiter: str = " ",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for laz_file in tqdm(files, desc="Converting LAZ -> XYZ"):
        las = laspy.read(str(laz_file))
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        out_path = output_dir / f"{laz_file.stem}.xyz"
        np.savetxt(out_path, xyz, fmt=xyz_fmt, delimiter=delimiter)

    print(f"Saved {len(files)} xyz files to: {output_dir}")


def visualize_sequence(
    files: list[Path],
    fps: float,
    loop: bool,
    use_intensity_color: bool,
) -> None:
    first = laz_to_pointcloud(files[0], use_intensity_color=use_intensity_color)
    if len(first.points) == 0:
        raise ValueError(f"First cloud is empty: {files[0]}")

    # ------------------------------------------------------------------ #
    #  Window 1 — original irregular cloud                                #
    # ------------------------------------------------------------------ #
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Irregular cloud (original)",
        width=960, height=540,
        left=0, top=40,
    )

    pcd = first
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)

    # ------------------------------------------------------------------ #
    #  Window 2 — regular cloud via spherical projection                  #
    # ------------------------------------------------------------------ #
    print("Building initial regular (spherical) cloud …", flush=True)
    reg_pcd = make_regular_spherical_cloud(first)
    print(
        f"  irregular: {len(first.points):,} pts  →  "
        f"regular: {len(reg_pcd.points):,} pts",
        flush=True,
    )

    # --- Save range image PNG (for report) --------------------------------
    range_img_path = files[0].parent / "range_image.png"
    save_range_image(first, out_path=range_img_path)

    vis_reg = o3d.visualization.Visualizer()
    vis_reg.create_window(
        window_name="Regular cloud (spherical projection)",
        width=960, height=540,
        left=960, top=40,         # side by side with window 1
    )
    vis_reg.add_geometry(reg_pcd)
    vis_reg.update_geometry(reg_pcd)
    vis_reg.poll_events()
    vis_reg.update_renderer()
    vis_reg.reset_view_point(True)

    # ------------------------------------------------------------------ #
    #  Shared state & key callbacks (only on window 1)                    #
    # ------------------------------------------------------------------ #
    state = {"running": True, "zoom": 0.7}
    vis.get_view_control().set_zoom(state["zoom"])

    def zoom_in(_: o3d.visualization.Visualizer) -> bool:
        state["zoom"] = max(0.02, state["zoom"] - 0.05)
        vis.get_view_control().set_zoom(state["zoom"])
        print(f"zoom: {state['zoom']:.2f}")
        return False

    def zoom_out(_: o3d.visualization.Visualizer) -> bool:
        state["zoom"] = min(2.0, state["zoom"] + 0.05)
        vis.get_view_control().set_zoom(state["zoom"])
        print(f"zoom: {state['zoom']:.2f}")
        return False

    def quit_viewer(_: o3d.visualization.Visualizer) -> bool:
        state["running"] = False
        return False

    vis.register_key_callback(ord("="), zoom_in)
    vis.register_key_callback(ord("+"), zoom_in)
    vis.register_key_callback(ord("-"), zoom_out)
    vis.register_key_callback(ord("_"), zoom_out)
    vis.register_key_callback(334, zoom_in)   # numpad +
    vis.register_key_callback(333, zoom_out)  # numpad -
    vis.register_key_callback(ord("Q"), quit_viewer)
    vis.register_key_callback(ord("q"), quit_viewer)

    print("Controls: '+' / '=' zoom in, '-' zoom out, 'q' quit.")

    frame_delay = 1.0 / fps if fps > 0 else 0.0
    reg_alive = True   # track whether window 2 is still open

    # ------------------------------------------------------------------ #
    #  Main render loop                                                    #
    # ------------------------------------------------------------------ #
    try:
        first_frame = True
        while state["running"]:
            for idx, laz_file in enumerate(files, start=1):
                if not state["running"]:
                    break

                if first_frame:
                    first_frame = False
                else:
                    current = laz_to_pointcloud(laz_file, use_intensity_color=use_intensity_color)
                    if len(current.points) == 0:
                        print(f"[skip] {laz_file.name} is empty")
                        continue

                    # Update irregular cloud in-place
                    pcd.points = current.points
                    pcd.colors = current.colors

                    # Recompute and update regular cloud in-place
                    if reg_alive:
                        current_reg = make_regular_spherical_cloud(current)
                        reg_pcd.points = current_reg.points
                        reg_pcd.colors = current_reg.colors

                # Refresh window 1
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                # Refresh window 2 (gracefully handle user closing it)
                if reg_alive:
                    try:
                        vis_reg.update_geometry(reg_pcd)
                        still_open = vis_reg.poll_events()
                        vis_reg.update_renderer()
                        if not still_open:
                            reg_alive = False
                    except Exception:
                        reg_alive = False

                print(f"[{idx}/{len(files)}] {laz_file.name}")
                if frame_delay > 0:
                    time.sleep(frame_delay)

            if not loop:
                break
    finally:
        vis.destroy_window()
        if reg_alive:
            try:
                vis_reg.destroy_window()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show .laz sequence or convert laz -> xyz."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder with .laz files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for switching clouds (default: 5).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over files continuously.",
    )
    parser.add_argument(
        "--no-intensity-color",
        action="store_true",
        help="Disable colorization by intensity.",
    )
    parser.add_argument(
        "--convert-to-xyz",
        action="store_true",
        help="Convert all .laz files from folder to .xyz and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for .xyz files (default: <folder>/xyz).",
    )
    parser.add_argument(
        "--xyz-fmt",
        type=str,
        default="%.6f",
        help="Floating point format for xyz export (default: %%.6f).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=" ",
        help="Delimiter for xyz export (default: space).",
    )
    parser.add_argument(
        "--h-res",
        type=float,
        default=0.2,
        help="Horizontal (azimuth) angular resolution for spherical grid, degrees (default: 0.2).",
    )
    parser.add_argument(
        "--v-res",
        type=float,
        default=0.2,
        help="Vertical (elevation) angular resolution for spherical grid, degrees (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")

    laz_files = collect_laz_files(folder)
    if args.convert_to_xyz:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else folder / "xyz"
        )
        convert_laz_folder_to_xyz(
            files=laz_files,
            output_dir=output_dir,
            xyz_fmt=args.xyz_fmt,
            delimiter=args.delimiter,
        )
        return

    visualize_sequence(
        files=laz_files,
        fps=args.fps,
        loop=args.loop,
        use_intensity_color=not args.no_intensity_color,
    )


if __name__ == "__main__":
    main()