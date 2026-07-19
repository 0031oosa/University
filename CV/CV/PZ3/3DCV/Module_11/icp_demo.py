#!/usr/bin/env python3
"""
Демонстрация алгоритма ICP (Iterative Closest Point).

Синтетическое 2D-облако: «эталон» и «скан» с известным сдвигом и поворотом.
Визуализация показывает шаги: ближайшие пары → оценка жёсткого преобразования →
применение → повтор до сходимости.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


@dataclass
class ICPStep:
    """Состояние одной итерации ICP."""
    iteration: int
    source: np.ndarray  # (N, 2) текущее положение «скана»
    correspondences: np.ndarray  # (N, 2) ближайшие точки на эталоне
    R: np.ndarray  # (2, 2)
    t: np.ndarray  # (2,)
    mean_error: float


def make_L_shape(n_per_segment: int = 40, noise: float = 0.008) -> np.ndarray:
    """L-образный контур — хорошо виден поворот при выравнивании."""
    seg1 = np.linspace([0.0, 0.0], [1.0, 0.0], n_per_segment)
    seg2 = np.linspace([1.0, 0.0], [1.0, 0.7], n_per_segment)
    pts = np.vstack([seg1, seg2[1:]])
    pts += np.random.default_rng(0).normal(0, noise, pts.shape)
    return pts


def rigid_transform_2d(points: np.ndarray, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = np.array([tx, ty])
    return (R @ points.T).T + t


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Жёсткое преобразование 2D: минимизация ||dst - (R @ src + t)||^2 (метод Кабша / SVD).
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_c = src - centroid_src
    dst_c = dst - centroid_dst
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def icp(
    source_init: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[ICPStep]]:
    """
    Point-to-point ICP: на каждом шаге для каждой точки скана ищем ближайшую на эталоне,
    оцениваем R, t и обновляем скан.
    """
    tree = cKDTree(target)
    source = source_init.copy()
    history: list[ICPStep] = []
    cumulative_R = np.eye(2)
    cumulative_t = np.zeros(2)

    for it in range(max_iterations):
        dists, indices = tree.query(source)
        correspondences = target[indices]
        mean_error = float(np.mean(dists**2))

        R_step, t_step = estimate_rigid_transform(source, correspondences)
        source = apply_transform(source, R_step, t_step)
        cumulative_R = R_step @ cumulative_R
        cumulative_t = R_step @ cumulative_t + t_step

        history.append(
            ICPStep(
                iteration=it,
                source=source.copy(),
                correspondences=correspondences.copy(),
                R=R_step.copy(),
                t=t_step.copy(),
                mean_error=mean_error,
            )
        )

        if it > 0 and abs(history[-2].mean_error - mean_error) < tolerance:
            break

    return cumulative_R, cumulative_t, history


def setup_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )


def plot_overview(
    target: np.ndarray,
    scan_initial: np.ndarray,
    scan_final: np.ndarray,
    true_R: np.ndarray,
    true_t: np.ndarray,
    est_R: np.ndarray,
    est_t: np.ndarray,
) -> plt.Figure:
    """Сводная схема: постановка задачи и результат."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(
        "ICP: выровнять «скан» (красный) с «эталоном» (синий), не зная истинного сдвига",
        fontsize=12,
        y=1.02,
    )

    panels = [
        (
            axes[0],
            "1. Два облака\n(скан смещён и повёрнут)",
            scan_initial,
            "до ICP",
        ),
        (
            axes[1],
            "2. Цель ICP\nминимизировать расстояния до эталона",
            scan_initial,
            "задача",
        ),
        (
            axes[2],
            "3. После ICP\nскан совпал с эталоном",
            scan_final,
            "после ICP",
        ),
    ]

    for ax, title, scan, _ in panels:
        ax.scatter(target[:, 0], target[:, 1], s=28, c="#2166ac", alpha=0.85, label="эталон (target)")
        ax.scatter(scan[:, 0], scan[:, 1], s=28, c="#b2182b", alpha=0.85, label="скан (source)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    # Стрелка «намерение» на средней панели
    ax = axes[1]
    c_tgt = target.mean(axis=0)
    c_src = scan_initial.mean(axis=0)
    ax.annotate(
        "",
        xy=c_tgt,
        xytext=c_src,
        arrowprops=dict(arrowstyle="->", color="gray", lw=2),
    )
    ax.text(
        0.5 * (c_src[0] + c_tgt[0]),
        0.5 * (c_src[1] + c_tgt[1]) + 0.05,
        "искомое\nпреобразование",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    # Ошибка оценки преобразования (для отчёта)
    angle_true = np.rad2deg(np.arctan2(true_R[1, 0], true_R[0, 0]))
    angle_est = np.rad2deg(np.arctan2(est_R[1, 0], est_R[0, 0]))
    fig.text(
        0.5,
        -0.02,
        f"Истина: поворот {angle_true:.1f}°, сдвиг ({true_t[0]:.2f}, {true_t[1]:.2f})  |  "
        f"ICP: поворот {angle_est:.1f}°, сдвиг ({est_t[0]:.2f}, {est_t[1]:.2f})",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    return fig


def plot_iterations(history: list[ICPStep], target: np.ndarray, source_init: np.ndarray) -> plt.Figure:
    """Сетка итераций: линии соответствий + накопленная ошибка."""
    n_show = min(6, len(history))
    pick = np.linspace(0, len(history) - 1, n_show, dtype=int)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, n_show, height_ratios=[1.2, 0.35], hspace=0.35, wspace=0.25)
    fig.suptitle(
        "Итерации ICP: (1) найти пары «ближайший сосед»  →  (2) оценить R, t  →  (3) сдвинуть скан",
        fontsize=12,
        y=0.98,
    )

    errors = [h.mean_error for h in history]
    iters = [h.iteration for h in history]

    for col, idx in enumerate(pick):
        step = history[idx]
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(target[:, 0], target[:, 1], s=20, c="#2166ac", alpha=0.5, zorder=1)
        ax.scatter(step.source[:, 0], step.source[:, 1], s=22, c="#b2182b", alpha=0.9, zorder=3)

        # Линии соответствий (ключевая идея ICP)
        for p, q in zip(step.source, step.correspondences):
            ax.plot([p[0], q[0]], [p[1], q[1]], c="#4daf4a", alpha=0.35, lw=0.8, zorder=2)

        ax.set_aspect("equal")
        ax.set_title(f"итерация {step.iteration}\nMSE = {step.mean_error:.5f}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("y")

    # График сходимости
    ax_err = fig.add_subplot(gs[1, :])
    ax_err.plot(iters, errors, "o-", color="#2166ac", lw=2, markersize=4)
    ax_err.set_xlabel("номер итерации")
    ax_err.set_ylabel("средний кв. расстояние\n(ошибка соответствий)")
    ax_err.set_title("Сходимость: ошибка падает, пока скан не совпадёт с эталоном")
    ax_err.grid(True, alpha=0.3)
    for idx in pick:
        ax_err.axvline(history[idx].iteration, color="#b2182b", alpha=0.2, lw=1)

    # Легенда
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac", markersize=8, label="эталон"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b", markersize=8, label="скан"),
        Line2D([0], [0], color="#4daf4a", lw=1.5, label="пара «скан → ближайший на эталоне»"),
    ]
    ax_err.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=9)

    return fig


def plot_one_step_detail(
    history: list[ICPStep],
    target: np.ndarray,
    source_init: np.ndarray,
    step_index: int = 0,
) -> plt.Figure:
    """Одна итерация «крупным планом»: центроиды и поворот."""
    step = history[step_index]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Разбор итерации {step.iteration} (один шаг ICP)", fontsize=12, y=1.02)

    # A: соответствия
    ax = axes[0]
    ax.scatter(target[:, 0], target[:, 1], s=30, c="#2166ac", label="эталон")
    ax.scatter(step.source[:, 0], step.source[:, 1], s=30, c="#b2182b", label="скан")
    for p, q in zip(step.source, step.correspondences):
        ax.plot([p[0], q[0]], [p[1], q[1]], "g-", alpha=0.4, lw=1)
    ax.set_title("Шаг A: для каждой точки скана\nнайти ближайшую на эталоне")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # B: центроиды
    ax = axes[1]
    src_c = step.source.mean(axis=0)
    dst_c = step.correspondences.mean(axis=0)
    ax.scatter(step.source[:, 0], step.source[:, 1], s=25, c="#b2182b", alpha=0.6)
    ax.scatter(step.correspondences[:, 0], step.correspondences[:, 1], s=25, c="#4daf4a", alpha=0.6)
    ax.scatter(*src_c, s=120, c="#b2182b", edgecolors="k", zorder=5, label="центр скана")
    ax.scatter(*dst_c, s=120, c="#4daf4a", edgecolors="k", zorder=5, label="центр пар")
    ax.annotate("", xy=dst_c, xytext=src_c, arrowprops=dict(arrowstyle="->", color="k", lw=2))
    ax.set_title("Шаг B: по парам оценить\nповорот R и сдвиг t (SVD)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # C: после применения шага (следующее состояние или текущее уже обновлённое в history)
    ax = axes[2]
    ax.scatter(target[:, 0], target[:, 1], s=30, c="#2166ac", alpha=0.7, label="эталон")
    if step_index + 1 < len(history):
        nxt = history[step_index + 1].source
        title = "Шаг C: применить R, t →\nскан ближе к эталону"
    else:
        nxt = step.source
        title = "Шаг C: финальное положение"
    ax.scatter(nxt[:, 0], nxt[:, 1], s=30, c="#b2182b", label="скан после шага")
    # Дуга поворота вокруг центра
    angle = np.arctan2(step.R[1, 0], step.R[0, 0])
    arc = patches.Arc(
        src_c,
        0.35,
        0.35,
        angle=0,
        theta1=0,
        theta2=np.rad2deg(angle),
        color="#984ea3",
        lw=2,
    )
    ax.add_patch(arc)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_algorithm_flowchart() -> plt.Figure:
    """Блок-схема алгоритма (текстовая)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Схема алгоритма ICP", fontsize=13, pad=12)

    boxes = [
        (5, 9.0, "Старт: облако «скан» S, эталон T"),
        (5, 7.5, "i = 0"),
        (5, 6.0, "Для каждой точки s ∈ S:\nнайти ближайшую t ∈ T"),
        (5, 4.3, "Оценить жёсткое (R, t):\nминимизировать Σ||t − (R·s + τ)||²"),
        (5, 2.6, "S ← R·S + τ\n(применить преобразование)"),
        (5, 1.0, "Ошибка уменьшилась?\nНет → i++, повтор\nДа → стоп, вернуть (R, t)"),
    ]
    for x, y, text in boxes:
        bbox = dict(boxstyle="round,pad=0.4", facecolor="#e8f4fc", edgecolor="#2166ac")
        ax.text(x, y, text, ha="center", va="center", fontsize=10, bbox=bbox)

    for y_from, y_to in [(9.0, 7.5), (7.5, 6.0), (6.0, 4.3), (4.3, 2.6), (2.6, 1.0)]:
        ax.annotate("", xy=(5, y_to + 0.35), xytext=(5, y_from - 0.35), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.annotate(
        "",
        xy=(8.2, 6.0),
        xytext=(8.2, 2.6),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="#b2182b", lw=1.5),
    )
    ax.text(8.6, 4.2, "нет", color="#b2182b", fontsize=10)
    ax.annotate(
        "",
        xy=(6.5, 7.5),
        xytext=(8.2, 6.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#b2182b", lw=1.2),
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Демонстрация ICP с визуализацией")
    parser.add_argument("--save", type=str, default="", help="префикс файлов для сохранения PNG (без расширения)")
    parser.add_argument("--no-show", action="store_true", help="не открывать окна matplotlib")
    args = parser.parse_args()

    setup_figure_style()
    rng = np.random.default_rng(42)

    # Эталон
    target = make_L_shape()
    # Истинное преобразование «скана» относительно эталона
    angle_true = 28.0
    tx_true, ty_true = 0.45, 0.35
    theta = np.deg2rad(angle_true)
    true_R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    true_t = np.array([tx_true, ty_true])

    source_gt = rigid_transform_2d(target, angle_true, tx_true, ty_true)
    source_init = source_gt + rng.normal(0, 0.01, source_gt.shape)

    est_R, est_t, history = icp(source_init, target)
    # Накопленное: проще взять финальный скан
    scan_final = history[-1].source

    fig1 = plot_overview(target, source_init, scan_final, true_R, true_t, est_R, est_t)
    fig2 = plot_iterations(history, target, source_init)
    fig3 = plot_one_step_detail(history, target, source_init, step_index=0)
    fig4 = plot_algorithm_flowchart()

    if args.save:
        prefix = args.save.rstrip("/")
        fig1.savefig(f"{prefix}_overview.png", bbox_inches="tight")
        fig2.savefig(f"{prefix}_iterations.png", bbox_inches="tight")
        fig3.savefig(f"{prefix}_step_detail.png", bbox_inches="tight")
        fig4.savefig(f"{prefix}_flowchart.png", bbox_inches="tight")
        print(f"Сохранено: {prefix}_*.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
