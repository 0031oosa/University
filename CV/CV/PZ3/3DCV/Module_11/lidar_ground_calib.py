#!/usr/bin/env python3
"""
Калибровка установки лидара по плоскости земли.

Запуск:
  python lidar_ground_calib.py /path/to/scan.laz

В консоль выводятся высота, pitch и roll; в каталог запуска пишется
lidar_ground_calib.json; открывается окно с облаком и плоскостью земли.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

# Лидар на крыше: ближняя зона ~3 м — кузов и слепая зона; RANSAC только от 10 м.
MIN_GROUND_RANGE_M = 10.0
# Пишется в каталог, из которого запущен скрипт (cwd).
OUTPUT_JSON_NAME = "lidar_ground_calib.json"


@dataclass
class LidarMountCalibration:
    height_m: float
    pitch_deg: float
    roll_deg: float
    plane_rmse_m: float
    num_ground_points: int

    def summary_ru(self) -> str:
        return (
            f"Высота над землёй:  {self.height_m:.4f} м\n"
            f"Тангаж (pitch):     {self.pitch_deg:+.3f}°\n"
            f"Крен (roll):        {self.roll_deg:+.3f}°\n"
            f"RMSE до плоскости:  {self.plane_rmse_m:.4f} м\n"
            f"Точек земли:        {self.num_ground_points}"
        )


def load_laz_pointcloud(laz_path: Path) -> o3d.geometry.PointCloud:
    import laspy

    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def auto_voxel_size(pcd: o3d.geometry.PointCloud, target_points: int = 80_000) -> float:
    pts = np.asarray(pcd.points)
    if len(pts) < target_points:
        return 0.0
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    volume = max(diag**3, 1e-9)
    voxel = (volume / target_points) ** (1.0 / 3.0)
    return float(np.clip(voxel, 0.05, 2.0))


def auto_ransac_distance(pcd: o3d.geometry.PointCloud) -> float:
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    median = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 0.1
    return float(np.clip(2.5 * median, 0.08, 0.35))


def ground_candidate_mask(points: np.ndarray) -> np.ndarray:
    """
    Кандидаты в землю: низкие по z, горизонтальная дальность >= 10 м.

    Ближе ~3 м — слепая зона и отражения от машины; в RANSAC их не берём.
    """
    z = points[:, 2]
    z_cut = np.percentile(z, 30)
    r_xy = np.hypot(points[:, 0], points[:, 1])
    return (z <= z_cut) & (r_xy >= MIN_GROUND_RANGE_M)


def orient_ground_normal_upward(normal: np.ndarray, ground_points: np.ndarray) -> np.ndarray:
    """Нормаль вверх (nz > 0); датчик над землёй → для земли n·p < 0."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    if n[2] < 0:
        n = -n
    if float(np.median(ground_points @ n)) > 0:
        n = -n
    return n


def fit_ground_plane(
    points: np.ndarray,
    distance_threshold: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    mask = ground_candidate_mask(points)
    if int(mask.sum()) < 500:
        raise RuntimeError(
            "Слишком мало точек-кандидатов в землю. Проверьте сцену или ориентацию осей."
        )

    candidates = o3d.geometry.PointCloud()
    candidates.points = o3d.utility.Vector3dVector(points[mask])

    coeffs, local_inliers = candidates.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000,
    )
    normal = np.asarray(coeffs[:3], dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    if abs(normal[2]) < 0.7:
        raise RuntimeError(
            f"Плоскость земли не горизонтальна: |nz|={abs(normal[2]):.3f}"
        )

    global_inliers = np.where(mask)[0][local_inliers]
    gnd = points[global_inliers]
    normal = orient_ground_normal_upward(normal, gnd)
    # Расстояние от лидара (начала координат) до плоскости вдоль нормали
    height = -float(np.median(gnd @ normal))
    if height <= 0:
        raise RuntimeError(
            f"Высота над землёй неположительна ({height:.3f} м). "
            "Возможно, неверная ориентация осей (ожидается z вверх)."
        )
    return normal, height, global_inliers


def pitch_roll_from_ground_normal(normal: np.ndarray) -> tuple[float, float]:
    """R = Ry(pitch) @ Rx(roll): выравнивание нормали земли с (0, 0, 1)."""
    nx, ny, nz = normal / (np.linalg.norm(normal) + 1e-12)
    roll = float(np.arctan2(ny, nz))
    pitch = float(np.arctan2(-nx, np.hypot(ny, nz)))
    return pitch, roll


def calibrate(
    ground_normal: np.ndarray,
    height_m: float,
    ground_points: np.ndarray,
) -> LidarMountCalibration:
    pitch, roll = pitch_roll_from_ground_normal(ground_normal)
    residuals = ground_points @ ground_normal + height_m
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return LidarMountCalibration(
        height_m=height_m,
        pitch_deg=float(np.degrees(pitch)),
        roll_deg=float(np.degrees(roll)),
        plane_rmse_m=rmse,
        num_ground_points=len(ground_points),
    )


def save_calibration_json(
    calib: LidarMountCalibration,
    ground_normal: np.ndarray,
    laz_path: Path,
    output_path: Path,
) -> None:
    pitch_rad = float(np.radians(calib.pitch_deg))
    roll_rad = float(np.radians(calib.roll_deg))
    payload = {
        **asdict(calib),
        "pitch_rad": pitch_rad,
        "roll_rad": roll_rad,
        "ground_normal_lidar": [float(x) for x in ground_normal],
        "source_laz": str(laz_path),
        "vehicle_frame": {
            "axes": "x_forward_y_left_z_up",
            "ground_plane_z": 0.0,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def colored_pointcloud(points: np.ndarray, ground_inliers: np.ndarray) -> o3d.geometry.PointCloud:
    colors = np.tile([0.55, 0.55, 0.55], (len(points), 1))
    colors[ground_inliers] = [0.15, 0.85, 0.25]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def ground_plane_mesh(
    normal: np.ndarray,
    height_m: float,
    ground_points: np.ndarray,
    margin: float = 1.1,
) -> o3d.geometry.TriangleMesh:
    """Плоскость n·p = height (проходит на расстоянии height под датчиком, не через него)."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    anchor = -n * height_m
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)

    rel = ground_points - anchor
    half_u = max(float(np.max(np.abs(rel @ u)) * margin), 3.0)
    half_v = max(float(np.max(np.abs(rel @ v)) * margin), 3.0)

    corners = [
        anchor - half_u * u - half_v * v,
        anchor + half_u * u - half_v * v,
        anchor + half_u * u + half_v * v,
        anchor - half_u * u + half_v * v,
    ]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.paint_uniform_color([0.95, 0.75, 0.15])
    mesh.compute_vertex_normals()
    return mesh


def visualize(
    points: np.ndarray,
    ground_inliers: np.ndarray,
    normal: np.ndarray,
    height_m: float,
) -> None:
    pcd = colored_pointcloud(points, ground_inliers)
    plane = ground_plane_mesh(normal, height_m, points[ground_inliers])
    o3d.visualization.draw_geometries(
        [pcd, plane],
        window_name="Лидар: серая — облако, зелёная — земля, жёлтая — плоскость",
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Использование: python lidar_ground_calib.py /path/to/scan.laz")
        sys.exit(1)

    laz_path = Path(sys.argv[1]).expanduser().resolve()
    if not laz_path.is_file() or laz_path.suffix.lower() != ".laz":
        raise FileNotFoundError(f"Нужен существующий файл .laz: {laz_path}")

    print(f"Файл: {laz_path.name}")
    pcd = load_laz_pointcloud(laz_path)
    print(f"Точек: {len(pcd.points)}")

    voxel = auto_voxel_size(pcd)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        print(f"После прореживания: {len(pcd.points)} точек")

    points = np.asarray(pcd.points)
    normal, height_m, inliers = fit_ground_plane(points, auto_ransac_distance(pcd))
    calib = calibrate(normal, height_m, points[inliers])

    print("\n--- Параметры установки лидара ---")
    print(calib.summary_ru())
    print(f"\nНормаль земли (СК лидара): {tuple(round(x, 5) for x in normal)}")

    json_path = Path.cwd() / OUTPUT_JSON_NAME
    save_calibration_json(calib, normal, laz_path, json_path)
    print(f"\nJSON: {json_path.resolve()}")

    visualize(points, inliers, normal, height_m)


if __name__ == "__main__":
    main()
