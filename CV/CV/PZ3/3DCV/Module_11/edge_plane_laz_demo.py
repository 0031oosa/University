#!/usr/bin/env python3
"""
Поиск плоских и краевых точек в облаке LAZ.

Режим hybrid (по умолчанию): RANSAC находит крупные плоскости (дорога, стены),
PCA по кривизне — только настоящие границы. Линейность скана не считается краем.

Запуск (из Module_11, venv активирован):
  python edge_plane_laz_demo.py /Users/rinat/testdata/lidar
  python edge_plane_laz_demo.py /path/scan.laz --voxel-size 0.12 --save-figures
"""
from __future__ import annotations

import argparse
from enum import IntEnum
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Ellipse

class PointLabel(IntEnum):
    OTHER = 0
    PLANE = 1
    EDGE = 2


LABEL_COLORS = {
    PointLabel.OTHER: np.array([0.55, 0.55, 0.55]),
    PointLabel.PLANE: np.array([0.15, 0.85, 0.25]),
    PointLabel.EDGE: np.array([0.95, 0.20, 0.15]),
}

LABEL_NAMES = {
    PointLabel.OTHER: "прочие",
    PointLabel.PLANE: "плоскость",
    PointLabel.EDGE: "край",
}


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


def compute_pca_features(
    pcd: o3d.geometry.PointCloud,
    knn: int,
    radius: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if radius is not None:
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn)
    else:
        search = o3d.geometry.KDTreeSearchParamKNN(knn=knn)

    covs = np.asarray(o3d.geometry.PointCloud.estimate_point_covariances(pcd, search))
    evals = np.linalg.eigvalsh(covs)
    evals = np.sort(evals, axis=1)
    l0, l1, l2 = evals[:, 0], evals[:, 1], evals[:, 2]
    scale = l2 + 1e-12

    planarity = (l1 - l0) / scale
    linearity = (l2 - l1) / scale
    curvature = l0 / (l0 + l1 + l2 + 1e-12)
    return planarity, linearity, curvature, evals


def classify_points_pca(
    planarity: np.ndarray,
    linearity: np.ndarray,
    curvature: np.ndarray,
    plane_plan_pct: float,
    plane_curv_pct: float,
    edge_curv_pct: float,
    edge_plan_max_pct: float,
    use_linearity_for_edge: bool,
    edge_lin_pct: float,
) -> np.ndarray:
    """
    PCA-классификация. Для лидара linearity обычно НЕ использовать:
    линии скана выглядят как «ребро», хотя геометрия плоская.
    """
    p_thr = np.percentile(planarity, plane_plan_pct)
    c_lo = np.percentile(curvature, plane_curv_pct)
    c_hi = np.percentile(curvature, edge_curv_pct)
    p_edge_max = np.percentile(planarity, edge_plan_max_pct)

    labels = np.full(planarity.shape[0], PointLabel.OTHER, dtype=np.int8)
    plane_mask = (planarity >= p_thr) & (curvature <= c_lo)
    labels[plane_mask] = PointLabel.PLANE

    edge_mask = (curvature >= c_hi) & (planarity <= p_edge_max) & (~plane_mask)
    if use_linearity_for_edge:
        l_thr = np.percentile(linearity, edge_lin_pct)
        edge_mask |= (linearity >= l_thr) & (~plane_mask)

    labels[edge_mask] = PointLabel.EDGE
    return labels


def segment_plane_labels(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    num_iterations: int,
    max_planes: int,
    min_plane_points: int,
) -> np.ndarray:
    """RANSAC: крупные плоские участки (дорога, стены, крыши)."""
    n = len(pcd.points)
    labels = np.full(n, PointLabel.OTHER, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)

    for _ in range(max_planes):
        if int(remaining.sum()) < min_plane_points:
            break
        sub = pcd.select_by_index(np.where(remaining)[0])
        _, inliers = sub.segment_plane(distance_threshold, 3, num_iterations)
        if len(inliers) < min_plane_points:
            break
        global_idx = np.where(remaining)[0][inliers]
        labels[global_idx] = PointLabel.PLANE
        remaining[global_idx] = False

    return labels


def classify_points_hybrid(
    pcd: o3d.geometry.PointCloud,
    planarity: np.ndarray,
    linearity: np.ndarray,
    curvature: np.ndarray,
    ransac_distance: float,
    ransac_iterations: int,
    max_planes: int,
    min_plane_points: int,
    edge_curv_pct: float,
    edge_plan_max_pct: float,
) -> np.ndarray:
    """RANSAC-плоскости + PCA-края только по кривизне на остальных точках."""
    labels = segment_plane_labels(
        pcd,
        distance_threshold=ransac_distance,
        num_iterations=ransac_iterations,
        max_planes=max_planes,
        min_plane_points=min_plane_points,
    )
    not_plane = labels != PointLabel.PLANE
    if not np.any(not_plane):
        return labels

    c_hi = np.percentile(curvature[not_plane], edge_curv_pct)
    p_max = np.percentile(planarity[not_plane], edge_plan_max_pct)
    edge_mask = not_plane & (curvature >= c_hi) & (planarity <= p_max)
    labels[edge_mask] = PointLabel.EDGE
    return labels


def auto_ransac_distance(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    if len(pts) < 2:
        return 0.15
    nn = min(30, len(pts) - 1)
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    median = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 0.1
    return float(np.clip(2.5 * median, 0.08, 0.35))


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float64)
    for label, color in LABEL_COLORS.items():
        colors[labels == label] = color
    return colors


def save_explanation_figures(
    out_dir: Path,
    planarity: np.ndarray,
    linearity: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
    evals: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_geometry_schematic(out_dir / "01_local_geometry.png")
    _save_eigenvalue_bars(out_dir / "02_eigenvalues_examples.png", evals, labels)
    _save_feature_scatter(out_dir / "03_planarity_vs_curvature.png", planarity, curvature, labels)
    _save_feature_histograms(
        out_dir / "04_feature_histograms.png",
        planarity,
        linearity,
        curvature,
        labels,
    )


def _save_geometry_schematic(path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [
        "Плоскость: λ₀ ≪ λ₁ ≈ λ₂",
        "Ребро: λ₀ ≪ λ₁ ≪ λ₂",
        "Угол: λ₀ ≈ λ₁ ≈ λ₂",
    ]
    radii = [(0.08, 0.35), (0.06, 0.45), (0.22, 0.22)]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    for ax, title, (rx, ry), color in zip(axes, titles, radii, colors):
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.scatter([0], [0], s=80, c="black", zorder=3)
        ax.add_patch(
            Ellipse((0, 0), width=2 * rx, height=2 * ry, angle=25, fill=False, lw=2, ec=color)
        )
        for r in np.linspace(0.12, 0.45, 5):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", lw=0.6, ec="#888888"))
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("локальная ось 1")
        if ax is axes[0]:
            ax.set_ylabel("локальная ось 2")

    fig.suptitle(
        "Форма эллипсоида ковариации в окрестности точки (схема)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_eigenvalue_bars(path: Path, evals: np.ndarray, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    examples = [
        (PointLabel.PLANE, "плоскость"),
        (PointLabel.EDGE, "край"),
        (PointLabel.OTHER, "прочие"),
    ]

    for ax, (label, name) in zip(axes, examples):
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            ax.text(0.5, 0.5, "нет точек", ha="center", va="center")
            ax.set_title(name)
            continue
        sample = evals[idx][np.random.default_rng(0).choice(idx.size, size=min(200, idx.size), replace=False)]
        mean_ev = np.mean(sample, axis=0)
        ax.bar(["λ₀", "λ₁", "λ₂"], mean_ev, color=["#4c72b0", "#55a868", "#c44e52"])
        ax.set_title(f"средние λ ({name})")
        ax.set_ylabel("значение")

    fig.suptitle("Собственные значения ковариации (среднее по классу)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_feature_scatter(
    path: Path,
    planarity: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
) -> None:
    rng = np.random.default_rng(1)
    n = planarity.shape[0]
    idx = rng.choice(n, size=min(40_000, n), replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label in PointLabel:
        mask = labels[idx] == label
        if not np.any(mask):
            continue
        ax.scatter(
            planarity[idx][mask],
            curvature[idx][mask],
            s=3,
            alpha=0.35,
            c=[LABEL_COLORS[label]],
            label=LABEL_NAMES[label],
        )
    ax.set_xlabel("планарность  (λ₁−λ₀)/λ₂")
    ax.set_ylabel("кривизна  λ₀/(λ₀+λ₁+λ₂)")
    ax.set_title("Признаки PCA: плоскости — верх-слева, края — низ-справа")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_feature_histograms(
    path: Path,
    planarity: np.ndarray,
    linearity: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    data = [
        (planarity, "планарность"),
        (linearity, "линейность"),
        (curvature, "кривизна"),
    ]

    for ax, (values, title) in zip(axes, data):
        for label in PointLabel:
            subset = values[labels == label]
            if subset.size == 0:
                continue
            ax.hist(
                subset,
                bins=60,
                alpha=0.55,
                density=True,
                label=LABEL_NAMES[label],
                color=LABEL_COLORS[label],
            )
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Распределение геометрических признаков по классам", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_class_stats(labels: np.ndarray, mode: str) -> None:
    total = labels.size
    print(f"\nКлассификация точек (режим: {mode}):")
    for label in PointLabel:
        count = int(np.sum(labels == label))
        print(f"  {LABEL_NAMES[label]:10s}: {count:8d}  ({100.0 * count / total:5.1f}%)")


def visualize_colored_cloud(pcd: o3d.geometry.PointCloud, window_name: str) -> None:
    print("\nОкно Open3D: вращение мышью, колёсико — масштаб, Q — выход.")
    print("Цвета: зелёный — плоскость, красный — край, серый — прочие.")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=1280,
        height=720,
    )


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    default_data = module_dir

    parser = argparse.ArgumentParser(
        description="Краевые и плоские точки в LAZ-облаке (PCA + раскраска)."
    )
    parser.add_argument(
        "laz_path",
        type=Path,
        nargs="?",
        default=default_data,
        help=f"Файл .laz или папка (по умолчанию: {default_data}).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Даунсэмплинг вокселями, м. Если не задан — подбирается автоматически.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "pca"),
        default="hybrid",
        help="hybrid: RANSAC-плоскости + PCA-края; pca: только PCA (default: hybrid).",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=50,
        help="Число соседей для локальной ковариации (default: 50).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.8,
        help="Радиус гибридного поиска соседей, м (default: 0.8).",
    )
    parser.add_argument(
        "--ransac-distance",
        type=float,
        default=None,
        help="Порог RANSAC до плоскости, м (default: авто по плотности).",
    )
    parser.add_argument(
        "--max-planes",
        type=int,
        default=4,
        help="Сколько плоскостей искать RANSAC (default: 4).",
    )
    parser.add_argument(
        "--use-linearity-edge",
        action="store_true",
        help="Считать краем высокую linearity (для лидара обычно даёт ложные края на дороге).",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Сохранить поясняющие графики в testdata/lidar/figures/.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Каталог для графиков (default: <laz_dir>/figures).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Не открывать окно Open3D (только статистика и графики).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    print("PCA по локальной ковариации...")
    planarity, linearity, curvature, evals = compute_pca_features(
        pcd, knn=args.knn, radius=args.radius
    )

    if args.mode == "hybrid":
        ransac_dist = (
            args.ransac_distance
            if args.ransac_distance is not None
            else auto_ransac_distance(pcd)
        )
        print(
            f"RANSAC: distance={ransac_dist:.3f} м, до {args.max_planes} плоскостей"
        )
        labels = classify_points_hybrid(
            pcd,
            planarity,
            linearity,
            curvature,
            ransac_distance=ransac_dist,
            ransac_iterations=2000,
            max_planes=args.max_planes,
            min_plane_points=max(500, len(pcd.points) // 200),
            edge_curv_pct=96.0,
            edge_plan_max_pct=75.0,
        )
    else:
        labels = classify_points_pca(
            planarity,
            linearity,
            curvature,
            plane_plan_pct=50.0,
            plane_curv_pct=55.0,
            edge_curv_pct=96.0,
            edge_plan_max_pct=80.0,
            use_linearity_for_edge=args.use_linearity_edge,
            edge_lin_pct=98.0,
        )

    print_class_stats(labels, args.mode)

    pcd.colors = o3d.utility.Vector3dVector(labels_to_colors(labels))

    if args.save_figures:
        fig_dir = (
            args.figures_dir.expanduser().resolve()
            if args.figures_dir is not None
            else laz_path.parent / "figures"
        )
        save_explanation_figures(fig_dir, planarity, linearity, curvature, labels, evals)
        print(f"\nГрафики сохранены: {fig_dir}")

    if not args.no_viewer:
        visualize_colored_cloud(pcd, window_name=f"Плоскости и края — {laz_path.name}")


if __name__ == "__main__":
    main()
