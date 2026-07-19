#!/usr/bin/env python3
"""
Построение NDT (Normal Distributions Transform) по лидарному облаку LAZ.

Каждая занятая ячейка сетки описывается гауссианой N(μ, Σ) по точкам внутри ячейки.
Такая карта используется для выравнивания сканов без явных соответствий точка–точка (в отличие от ICP).

Запуск (из Module_11, venv активирован):
  python ndt_laz_demo.py /Users/rinat/testdata/lidar
  python ndt_laz_demo.py /path/scan.laz --ndt-resolution 0.5 --save-figures
  python ndt_laz_demo.py /path/folder --align-demo
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation


@dataclass
class NDTCell:
    """Одна ячейка NDT-карты: нормальное распределение в вокселе."""

    voxel_idx: tuple[int, int, int]
    center: np.ndarray  # геометрический центр вокселя, (3,)
    mean: np.ndarray  # μ — среднее точек в ячейке, (3,)
    cov: np.ndarray  # Σ — ковариация + регуляризация, (3, 3)
    inv_cov: np.ndarray  # Σ⁻¹ для оценки правдоподобия
    num_points: int


@dataclass
class NDTMap:
    origin: np.ndarray
    resolution: float
    cells: list[NDTCell]

    @property
    def num_cells(self) -> int:
        return len(self.cells)


def resolve_laz_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".laz":
        return path
    if path.is_dir():
        laz_files = sorted(path.glob("*.laz"))
        if not laz_files:
            raise FileNotFoundError(f"В каталоге нет .laz: {path}")
        return laz_files[0]
    raise FileNotFoundError(f"Нужен файл .laz или папка с ними: {path}")


def load_laz_pointcloud(laz_path: Path) -> o3d.geometry.PointCloud:
    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def auto_voxel_size(pcd: o3d.geometry.PointCloud, target_points: int = 120_000) -> float:
    pts = np.asarray(pcd.points)
    if len(pts) < target_points:
        return 0.0
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    volume = max(diag**3, 1e-9)
    voxel = (volume / target_points) ** (1.0 / 3.0)
    return float(np.clip(voxel, 0.05, 2.0))


def auto_ndt_resolution(points: np.ndarray) -> float:
    """Размер ячейки NDT: ~2.5× медианное расстояние до ближайшего соседа."""
    if len(points) < 2:
        return 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    positive = dists[dists > 0]
    median = float(np.median(positive)) if positive.size else 0.15
    return float(np.clip(2.5 * median, 0.15, 1.5))


def points_to_voxel_indices(points: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    return np.floor((points - origin) / resolution).astype(np.int32)


def build_ndt_map(
    points: np.ndarray,
    resolution: float,
    min_points_per_cell: int,
    epsilon: float,
) -> NDTMap:
    """
    Построение NDT-карты:
      1) Оси сетки: origin = min(xyz), ячейка (i,j,k) покрывает куб resolution³.
      2) Точки группируются по индексу вокселя.
      3) В каждой ячейке с достаточным числом точек: μ = mean, Σ = cov + εI.
    """
    origin = points.min(axis=0)
    voxel_idx = points_to_voxel_indices(points, origin, resolution)

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(map(tuple, voxel_idx)):
        buckets[key].append(i)

    cells: list[NDTCell] = []
    eye = np.eye(3, dtype=np.float64)

    for key, indices in buckets.items():
        n = len(indices)
        if n < min_points_per_cell:
            continue

        pts = points[indices]
        pts_u = np.unique(pts, axis=0)
        mean = pts.mean(axis=0)
        iso = (resolution * 0.35) ** 2
        if pts_u.shape[0] == 1:
            cov = iso * eye + epsilon * eye
        elif pts_u.shape[0] == 2:
            cov = np.cov(pts_u.T, bias=True) + iso * eye + epsilon * eye
        else:
            cov = np.cov(pts_u.T, bias=True)
            if cov.shape != (3, 3):
                cov = np.atleast_2d(cov).astype(np.float64)
                if cov.size == 1:
                    cov = np.full((3, 3), float(cov.flat[0]))
            cov = cov + epsilon * eye

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        idx_arr = np.array(key, dtype=np.float64)
        center = origin + (idx_arr + 0.5) * resolution

        cells.append(
            NDTCell(
                voxel_idx=key,
                center=center,
                mean=mean,
                cov=cov,
                inv_cov=inv_cov,
                num_points=n,
            )
        )

    return NDTMap(origin=origin, resolution=resolution, cells=cells)


def ndt_score_points(map_: NDTMap, points: np.ndarray) -> float:
    """
    Суммарный лог-скор NDT (Magnusson): для каждой точки ищется ячейка,
    в которую она попадает; вклад exp(-½ (p-μ)ᵀ Σ⁻¹ (p-μ)).
    """
    if not map_.cells:
        return 0.0

    origin = map_.origin
    res = map_.resolution
    cell_lut: dict[tuple[int, int, int], NDTCell] = {c.voxel_idx: c for c in map_.cells}

    total = 0.0
    for p in points:
        key = tuple(points_to_voxel_indices(p.reshape(1, 3), origin, res)[0])
        cell = cell_lut.get(key)
        if cell is None:
            continue
        d = p - cell.mean
        mahal = float(d @ cell.inv_cov @ d)
        total += np.exp(-0.5 * mahal)
    return total


def se3_from_twist(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xi = [tx, ty, tz, rx, ry, rz] — малый вектор в осях Ли."""
    t = xi[:3]
    rot = Rotation.from_rotvec(xi[3:])
    return rot.as_matrix(), t


def apply_se3(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def align_with_ndt(
    source: np.ndarray,
    target_map: NDTMap,
    max_iterations: int = 40,
    learning_rate: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Упрощённый градиентный подъём скора NDT по SE(3).
    Для демонстрации: старт с небольшого случайного возмущения, восстановление позы.
    """
    xi = np.zeros(6, dtype=np.float64)
    scores: list[float] = []

    for _ in range(max_iterations):
        R, t = se3_from_twist(xi)
        transformed = apply_se3(source, R, t)
        score = ndt_score_points(target_map, transformed)
        scores.append(score)

        grad = np.zeros(6, dtype=np.float64)
        eps = 1e-3
        for k in range(6):
            xi_plus = xi.copy()
            xi_plus[k] += eps
            Rp, tp = se3_from_twist(xi_plus)
            sp = ndt_score_points(target_map, apply_se3(source, Rp, tp))
            grad[k] = (sp - score) / eps

        xi += learning_rate * grad

    R, t = se3_from_twist(xi)
    return R, t, scores


def cell_color_by_count(num_points: int, max_count: int) -> np.ndarray:
    if max_count <= 0:
        return np.array([0.5, 0.5, 0.5])
    t = np.clip(num_points / max_count, 0.0, 1.0)
    # низкая плотность — синий, высокая — оранжевый/красный
    return np.array([0.15 + 0.85 * t, 0.35 + 0.45 * t, 1.0 - 0.85 * t])


def color_points_by_ndt_cells(points: np.ndarray, ndt_map: NDTMap) -> np.ndarray:
    """Каждая точка в NDT-ячейке получает цвет по плотности ячейки; остальные — серые."""
    n = len(points)
    colors = np.full((n, 3), 0.42, dtype=np.float64)
    if not ndt_map.cells:
        return colors

    cell_lut = {c.voxel_idx: c for c in ndt_map.cells}
    max_count = max(c.num_points for c in ndt_map.cells)
    vidx = points_to_voxel_indices(points, ndt_map.origin, ndt_map.resolution)

    for i in range(n):
        cell = cell_lut.get(tuple(vidx[i]))
        if cell is not None:
            colors[i] = cell_color_by_count(cell.num_points, max_count)
    return colors


def focus_region_bbox(
    ndt_map: NDTMap,
    points: np.ndarray,
    top_cells: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ограничить вид на плотный фрагмент сцены (иначе при диагонали ~200 м
    ячейки 0.15 м не видны на общем плане).
    """
    total = len(points)
    ranked = sorted(ndt_map.cells, key=lambda c: c.num_points, reverse=True)
    # Ячейки с тысячами одинаковых координат — артефакт дубликатов, не для кадра.
    ranked = [c for c in ranked if c.num_points < max(500, total // 20)]
    ranked = ranked[:top_cells]
    if not ranked:
        ranked = ndt_map.cells[: min(top_cells, len(ndt_map.cells))]

    means = np.array([c.mean for c in ranked], dtype=np.float64)
    lo = means.min(axis=0) - margin
    hi = means.max(axis=0) + margin
    return lo, hi


def mask_in_bbox(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def display_cov_radii(cov: np.ndarray, resolution: float, n_sigma: float) -> np.ndarray:
    """Радиусы для отрисовки: не меньше половины вокселя (иначе невидимы)."""
    evals = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
    statistical = n_sigma * np.sqrt(evals)
    floor = 0.48 * resolution
    return np.maximum(statistical, floor)


def make_ellipsoid_mesh(
    center: np.ndarray,
    cov: np.ndarray,
    resolution: float,
    n_sigma: float,
    color: np.ndarray,
    resolution_mesh: int = 10,
) -> o3d.geometry.TriangleMesh:
    _, evecs = np.linalg.eigh(cov)
    radii = display_cov_radii(cov, resolution, n_sigma)

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution_mesh)
    transform = np.eye(4)
    transform[:3, :3] = evecs @ np.diag(radii)
    transform[:3, 3] = center
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color.tolist())
    return mesh


_CUBE_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def build_voxel_wireframes(
    cells: list[NDTCell],
    resolution: float,
    max_count: int,
) -> o3d.geometry.LineSet:
    """Каркасы вокселей r×r×r — хорошо читаются в 3D."""
    half = 0.5 * resolution
    offsets = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=np.float64,
    ) * half

    all_pts: list[np.ndarray] = []
    all_lines: list[list[int]] = []
    all_cols: list[np.ndarray] = []
    v_off = 0

    for cell in cells:
        corners = cell.center + offsets
        color = cell_color_by_count(cell.num_points, max_count)
        all_pts.append(corners)
        for e0, e1 in _CUBE_EDGES:
            all_lines.append([v_off + e0, v_off + e1])
            all_cols.append(color)
        v_off += 8

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(all_cols))
    return line_set


def visualize_ndt(
    points: np.ndarray,
    ndt_map: NDTMap,
    max_ellipsoids: int,
    n_sigma: float,
    window_name: str,
    focus_region: bool = True,
    focus_margin: float = 6.0,
    show_wireframes: bool = True,
    show_ellipsoids: bool = True,
    point_size: float = 2.5,
) -> None:
    if focus_region:
        lo, hi = focus_region_bbox(ndt_map, points, top_cells=500, margin=focus_margin)
        in_view = mask_in_bbox(points, lo, hi)
        pts_vis = points[in_view]
        span = hi - lo
        print(
            f"\nФокус вида: {span[0]:.1f}×{span[1]:.1f}×{span[2]:.1f} м "
            f"({int(in_view.sum())} точек из {len(points)})"
        )
    else:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        pts_vis = points

    colors = color_points_by_ndt_cells(pts_vis, ndt_map)
    n_colored = int(np.sum(np.any(colors < 0.41, axis=1)))
    print(f"  Точек в NDT-ячейках (цветные): {n_colored} / {len(pts_vis)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_vis)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    cells_in_view = [
        c for c in ndt_map.cells if np.all(c.mean >= lo) and np.all(c.mean <= hi)
    ]
    cells_sorted = sorted(cells_in_view, key=lambda c: c.num_points, reverse=True)
    show_cells = cells_sorted[:max_ellipsoids]
    max_count = max((c.num_points for c in show_cells), default=1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)

    marker_r = 0.55 * ndt_map.resolution
    for cell in show_cells[: min(250, len(show_cells))]:
        color = cell_color_by_count(cell.num_points, max_count)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_r, resolution=8)
        sphere.translate(cell.mean)
        sphere.paint_uniform_color(color.tolist())
        vis.add_geometry(sphere)

    if show_wireframes and show_cells:
        vis.add_geometry(build_voxel_wireframes(show_cells, ndt_map.resolution, max_count))

    if show_ellipsoids:
        for cell in show_cells[: min(120, len(show_cells))]:
            color = cell_color_by_count(cell.num_points, max_count)
            vis.add_geometry(
                make_ellipsoid_mesh(
                    cell.mean,
                    cell.cov,
                    ndt_map.resolution,
                    n_sigma=n_sigma,
                    color=color,
                )
            )

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.08, 0.08, 0.10])
    opt.mesh_show_back_face = True

    vis.reset_view_point(True)
    print(
        "\nОкно Open3D:"
        "\n  • цветные точки — попали в NDT-ячейку (синий→красный: плотнее ячейка)"
        "\n  • крупные маркеры — центры μ выбранных ячеек"
        "\n  • проволочные кубы — границы вокселя r×r×r"
        "\n  • полупрозрачные эллипсоиды — Σ (не меньше ~половины вокселя)"
        "\n  Вращение мышью; Q — выход."
    )
    vis.run()
    vis.destroy_window()


def print_build_stats(points: np.ndarray, ndt_map: NDTMap) -> None:
    counts = np.array([c.num_points for c in ndt_map.cells], dtype=np.int64)
    print("\n--- Построение NDT ---")
    print(f"  Точек в облаке:        {len(points)}")
    print(f"  Размер ячейки:         {ndt_map.resolution:.4f} м")
    print(f"  Занятых ячеек (≥ min): {ndt_map.num_cells}")
    if counts.size:
        print(f"  Точек в ячейке:        min={counts.min()}, median={np.median(counts):.0f}, max={counts.max()}")
    occupied_ratio = ndt_map.num_cells / max(len(points), 1)
    print(f"  Доля точек в «плотных» ячейках: ~{100 * occupied_ratio:.1f}% (грубо, 1 точка ≈ 1 ячейка max)")


def explain_process() -> None:
    print(
        """
=== Как строится NDT ===

1. Вокселизация
   Пространство режется на кубы со стороной r (ndt-resolution).
   Индекс ячейки: floor((p - origin) / r), origin = min(x,y,z).

2. Локальная гауссиана в каждой ячейке
   Для всех точек в одной ячейке:
     μ = (1/N) Σ pᵢ
     Σ = Cov(pᵢ) + ε·I     ← ε не даёт матрице выродиться на плоских стенах

3. Карта = набор {μ, Σ} только для ячеек с N ≥ min_points_per_cell
   Пустые ячейки не хранятся — карта разрежённая.

4. Смысл для регистрации (в отличие от ICP)
   Новый скан не ищет ближайшую точку, а оценивает, насколько он
   «правдоподобен» под каждой гауссианой: exp(-½ (p-μ)ᵀ Σ⁻¹ (p-μ)).
   Оптимизируется поза лидара, чтобы суммарный скор был максимален.

Флаг --align-demo: то же облако слегка сдвигается; градиентный подъём
восстанавливает позу по NDT-скору (учебный пример, не production SLAM).
"""
    )


def save_explanation_figures(
    out_dir: Path,
    points: np.ndarray,
    ndt_map: NDTMap,
    n_sigma: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_ndt_schematic(out_dir / "01_ndt_voxel_to_gaussian.png")
    _save_bev_slice(out_dir / "02_bev_points_and_ellipses.png", points, ndt_map, n_sigma)
    _save_cell_histogram(out_dir / "03_points_per_cell.png", ndt_map)


def _save_ndt_schematic(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.set_title("Шаг 1–2: точки в вокселе")
    ax.set_aspect("equal")
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.normal(0.35, 0.08, 35), rng.normal(0.35, 0.06, 35)])
    ax.scatter(pts[:, 0], pts[:, 1], s=25, c="#4472c4", edgecolors="k", linewidths=0.3)
    rect = plt.Rectangle((0.1, 0.1), 0.5, 0.5, fill=False, lw=2, ec="#333333")
    ax.add_patch(rect)
    ax.text(0.35, 0.05, "воксель r×r", ha="center", fontsize=10)
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.set_title("Шаг 3: гауссиана N(μ, Σ) в ячейке")
    ax.set_aspect("equal")
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T) + 0.002 * np.eye(2)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, c="#4472c4", alpha=0.6)
    ax.scatter([mu[0]], [mu[1]], s=120, c="#c00000", marker="x", linewidths=2, label="μ")
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    width, height = 2 * np.sqrt(evals)
    angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    ax.add_patch(
        Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            lw=2,
            ec="#c00000",
            label="1σ эллипс",
        )
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0.05, 0.65)
    ax.set_ylim(0.05, 0.65)
    ax.grid(True, alpha=0.25)

    fig.suptitle("От точек в ячейке к нормальному распределению (2D-схема)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_bev_slice(
    path: Path,
    points: np.ndarray,
    ndt_map: NDTMap,
    n_sigma: float,
    max_cells: int = 120,
) -> None:
    """Вид сверху (XY): точки и эллипсы Σ в плоскости XY."""
    rng = np.random.default_rng(2)
    n_show = min(25_000, len(points))
    idx = rng.choice(len(points), size=n_show, replace=False)
    pts = points[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="#888888", alpha=0.35, rasterized=True)

    cells = sorted(ndt_map.cells, key=lambda c: c.num_points, reverse=True)[:max_cells]
    res = ndt_map.resolution
    for cell in cells:
        mu = cell.mean[:2]
        cov2 = cell.cov[:2, :2]
        evals, evecs = np.linalg.eigh(cov2)
        radii2 = np.maximum(n_sigma * np.sqrt(np.maximum(evals, 1e-12)), 0.48 * res)
        w, h = 2 * radii2[0], 2 * radii2[1]
        ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        t = cell.num_points / max(c.num_points for c in cells)
        ax.add_patch(
            Ellipse(
                xy=mu,
                width=w,
                height=h,
                angle=ang,
                fill=False,
                lw=0.8,
                ec=plt.cm.plasma(0.2 + 0.75 * t),
                alpha=0.85,
            )
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_title(f"NDT сверху: до {max_cells} самых плотных ячеек (n_sigma={n_sigma})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_cell_histogram(path: Path, ndt_map: NDTMap) -> None:
    counts = np.array([c.num_points for c in ndt_map.cells], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 4))
    if counts.size:
        ax.hist(counts, bins=50, color="#4472c4", edgecolor="white", alpha=0.9)
    ax.set_xlabel("число точек в ячейке")
    ax.set_ylabel("число ячеек")
    ax.set_title("Распределение наполненности вокселей")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Построение NDT-карты по LAZ и визуализация гауссиан в ячейках."
    )
    parser.add_argument(
        "laz_path",
        type=Path,
        nargs="?",
        default=module_dir,
        help=f"Файл .laz или папка (по умолчанию: {module_dir}).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Даунсэмплинг облака, м (default: авто).",
    )
    parser.add_argument(
        "--ndt-resolution",
        type=float,
        default=None,
        help="Размер ячейки NDT r, м (default: авто по плотности).",
    )
    parser.add_argument(
        "--min-points-per-cell",
        type=int,
        default=6,
        help="Минимум точек в ячейке, чтобы хранить гауссиану (default: 6).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Регуляризация Σ ← Σ + εI (default: 1e-4).",
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=2.0,
        help="Масштаб эллипсоидов в визуализации: n_sigma * sqrt(λᵢ) (default: 2).",
    )
    parser.add_argument(
        "--max-ellipsoids",
        type=int,
        default=400,
        help="Сколько ячеек рисовать в Open3D (самые плотные) (default: 400).",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Сохранить поясняющие 2D-графики рядом с данными.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Каталог для графиков (default: <laz_dir>/ndt_figures).",
    )
    parser.add_argument(
        "--align-demo",
        action="store_true",
        help="Демо: сдвинуть облако и выровнять по NDT-скору (градиент).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Не открывать Open3D.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Вывести текстовое описание этапов построения NDT.",
    )
    parser.add_argument(
        "--full-scene",
        action="store_true",
        help="Не обрезать вид до плотного фрагмента (ячейки могут быть не видны).",
    )
    parser.add_argument(
        "--no-wireframes",
        action="store_true",
        help="Не рисовать каркасы вокселей.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.explain:
        explain_process()

    laz_path = resolve_laz_path(args.laz_path)
    print(f"LAZ: {laz_path.name}  ({laz_path.parent})")

    pcd = load_laz_pointcloud(laz_path)
    n_raw = len(pcd.points)
    print(f"Точек в файле: {n_raw}")

    voxel = args.voxel_size
    if voxel is None:
        voxel = auto_voxel_size(pcd)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        print(f"Даунсэмплинг: voxel={voxel:.3f} м → {len(pcd.points)} точек")

    points = np.asarray(pcd.points)
    resolution = args.ndt_resolution if args.ndt_resolution is not None else auto_ndt_resolution(points)
    print(f"NDT resolution: {resolution:.4f} м")

    print("Построение NDT-карты (группировка по вокселям → μ, Σ)...")
    ndt_map = build_ndt_map(
        points,
        resolution=resolution,
        min_points_per_cell=args.min_points_per_cell,
        epsilon=args.epsilon,
    )
    print_build_stats(points, ndt_map)

    if args.align_demo:
        rng = np.random.default_rng(42)
        xi_true = np.array([0.4, -0.25, 0.05, 0.02, -0.01, 0.03])
        R_true, t_true = se3_from_twist(xi_true)
        source = apply_se3(points, R_true, t_true)
        print("\n--- Демо выравнивания по NDT ---")
        print(f"  Истинный сдвиг (м):     {t_true}")
        print(f"  Истинный rotvec (рад):  {xi_true[3:]}")

        R_est, t_est, scores = align_with_ndt(source, ndt_map)
        print(f"  Оценённый сдвиг (м):    {t_est}")
        print(f"  Ошибка |Δt| (м):        {np.linalg.norm(t_est - t_true):.4f}")
        print(f"  NDT-скор: начало={scores[0]:.2f} → конец={scores[-1]:.2f}")

    if args.save_figures:
        fig_dir = (
            args.figures_dir.expanduser().resolve()
            if args.figures_dir is not None
            else laz_path.parent / "ndt_figures"
        )
        save_explanation_figures(fig_dir, points, ndt_map, args.n_sigma)
        print(f"\nГрафики: {fig_dir}")

    if not args.no_viewer:
        visualize_ndt(
            points,
            ndt_map,
            max_ellipsoids=args.max_ellipsoids,
            n_sigma=args.n_sigma,
            window_name=f"NDT — {laz_path.name}",
            focus_region=not args.full_scene,
            show_wireframes=not args.no_wireframes,
        )


if __name__ == "__main__":
    main()
