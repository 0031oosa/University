"""Графики по признакам уровня сущности из build_features.py — распределение
повторяемости, связности через общие площадки и связь между ними. Строится
на тех же агрегатах, что и entity_features_masked.csv, палитра — та же,
что и в visualize.py (см. references/palette.md skill'а dataviz —
валидирована, переиспользуется как есть).

Ни card, ни phone здесь не участвуют — только уже посчитанные
build_entity_features() агрегаты (occurrences, max_shared_casino_degree и
т.д.), поэтому график физически не может показать полный номер.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_features import build_entity_features, load_raw
from visualize import CATEGORICAL, CHROME, MUTED_GRAY

logger = logging.getLogger("analyze_features")

FEATURE_ASSETS_DIR = Path(__file__).parent / "feature_assets"

VALUE_KIND_COLORS = {"card": CATEGORICAL["blue"], "phone": CATEGORICAL["aqua"]}


def _style_axes(ax) -> None:
    ax.set_facecolor(CHROME["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHROME["baseline"])
    ax.spines["bottom"].set_color(CHROME["baseline"])
    ax.tick_params(colors=CHROME["text_secondary"])
    ax.grid(axis="y", color=CHROME["gridline"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_recurrence_distribution(features, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_occ = int(features["occurrences"].max()) if len(features) else 1
    bins = np.logspace(0, np.log10(max(max_occ, 2)), 30)
    for kind, label in [("card", "Карты"), ("phone", "Номера СБП")]:
        values = features.loc[features["value_kind"] == kind, "occurrences"]
        if values.empty:
            continue
        ax.hist(values, bins=bins, color=VALUE_KIND_COLORS[kind], alpha=0.65, label=label, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число повторных появлений одной и той же сущности")
    ax.set_ylabel("Число сущностей")
    ax.set_title("Распределение повторяемости card/phone-сущностей")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    path = out_dir / "fig1_recurrence_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_connectivity_distribution(features, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = features["max_shared_casino_degree"]
    nonzero = values[values > 0]
    if len(nonzero):
        bins = np.logspace(0, np.log10(max(nonzero.max(), 2)), 30)
        ax.hist(nonzero, bins=bins, color=CATEGORICAL["violet"], edgecolor="none")
        ax.set_xscale("log")
    zero_count = int((values == 0).sum())
    ax.set_yscale("log")
    ax.set_xlabel("Макс. число других сущностей на самой «многолюдной» площадке")
    ax.set_ylabel("Число сущностей")
    ax.set_title(
        f"Распределение связности через общие площадки\n"
        f"(без площадки/связей: {zero_count} из {len(values)} сущностей — не показаны на лог-шкале)",
        fontsize=11,
    )
    _style_axes(ax)
    fig.tight_layout()
    path = out_dir / "fig2_connectivity_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recurrence_vs_connectivity(features, out_dir: Path) -> Path:
    # Отдельные панели на card/phone, а не общий hexbin: fig1 уже показал, что
    # у карт и телефонов принципиально разный диапазон повторяемости (карты —
    # до ~40, телефоны — до ~300), общий hexbin эту разницу маскирует.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    titles = {"card": "Карты", "phone": "Номера СБП"}
    last_hb = None
    for ax, kind in zip(axes, ["card", "phone"]):
        subset = features[features["value_kind"] == kind]
        x = subset["occurrences"].clip(lower=1)
        y = subset["max_shared_casino_degree"].clip(lower=1)
        last_hb = ax.hexbin(x, y, gridsize=35, xscale="log", yscale="log", cmap="Blues", mincnt=1, linewidths=0.2)
        ax.set_xlabel("Число повторных появлений (occurrences)")
        ax.set_title(titles[kind], fontsize=11)
        _style_axes(ax)
    axes[0].set_ylabel("Связность через общие площадки\n(max_shared_casino_degree)")
    fig.colorbar(last_hb, ax=axes, label="Число сущностей (в ячейке)", fraction=0.05, pad=0.02)
    fig.suptitle("Повторяемость и связность через площадки — совместное распределение", y=1.02)
    path = out_dir / "fig3_recurrence_vs_connectivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_all_plots(features, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_recurrence_distribution(features, out_dir),
        plot_connectivity_distribution(features, out_dir),
        plot_recurrence_vs_connectivity(features, out_dir),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite",
        help="Путь(и) к SQLite-хранилищу через запятую",
    )
    parser.add_argument("--out-dir", default=str(FEATURE_ASSETS_DIR))
    args = parser.parse_args()

    raw = load_raw(args.db)
    features = build_entity_features(raw)
    logger.info(
        "Сущностей: %d (медиана повторений=%.0f, максимум=%d; медиана связности=%.0f, максимум=%d)",
        len(features),
        features["occurrences"].median() if len(features) else 0,
        features["occurrences"].max() if len(features) else 0,
        features["max_shared_casino_degree"].median() if len(features) else 0,
        features["max_shared_casino_degree"].max() if len(features) else 0,
    )
    paths = build_all_plots(features, Path(args.out_dir))
    for p in paths:
        logger.info("Сохранён график: %s", p)


if __name__ == "__main__":
    main()
