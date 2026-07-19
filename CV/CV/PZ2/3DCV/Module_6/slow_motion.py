"""
Slow motion с заполнением временной шкалы: выход всегда с заданным FPS (--output-fps),
длительность = длительность_исходника × factor. Между редкими ключевыми кадрами
(после --sample-fps / --stride) дорисовываются промежуточные кадры.

  --interp blend   — линейное смешивание соседних ключей (быстро, «двойные контуры»)
  --interp flow    — оптический поток Farneback между соседними ключами (медленнее, естественнее)
  --interp nearest — ближайший ключ без интерполяции (ступеньки, для сравнения)

Примеры:
  python slow_motion.py -i ../data/optical_flow.mp4 -f 4 --sample-fps 8 --interp blend
  python slow_motion.py -i ../data/optical_flow.mp4 -f 4 --sample-fps 8 --interp flow
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys

import cv2
import numpy as np


def open_writer(path: str, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if writer.isOpened():
        return writer, path
    alt = os.path.splitext(path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(alt, fourcc, float(fps), (w, h))
    if writer.isOpened():
        return writer, alt
    raise RuntimeError("Не удалось открыть VideoWriter; попробуйте другой путь или кодек.")


def collect_keyframes(
    cap: cv2.VideoCapture,
    file_fps: float,
    sample_fps: float | None,
    stride: int | None,
) -> list[tuple[float, np.ndarray]]:
    """Список (время_в_сек_на_шкале_исходника, кадр). Кадр = uint8 BGR."""
    out: list[tuple[float, np.ndarray]] = []
    idx = 0

    if stride is not None and stride > 1:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                out.append((idx / file_fps, frame.copy()))
            idx += 1
        return out

    if sample_fps is not None and sample_fps > 0 and sample_fps < file_fps - 1e-6:
        ratio = sample_fps / file_fps
        acc = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            acc += ratio
            while acc >= 1.0:
                out.append((idx / file_fps, frame.copy()))
                acc -= 1.0
            idx += 1
        return out

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.append((idx / file_fps, frame.copy()))
        idx += 1
    return out


def interp_nearest(
    keys: list[tuple[float, np.ndarray]],
    times: list[float],
    t_src: float,
) -> np.ndarray:
    i = bisect.bisect_right(times, t_src) - 1
    i = max(0, min(i, len(keys) - 1))
    return keys[i][1]


def interp_blend_pair(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)


def frame_at_time(
    keys: list[tuple[float, np.ndarray]],
    times: list[float],
    t_src: float,
    mode: str,
    flow_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None,
    xy: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """t_src в секундах по шкале исходника; flow_cache сегмент (i,i+1) -> (flow01, flow10)."""
    if not keys:
        raise ValueError("нет ключевых кадров")
    if len(keys) == 1:
        return keys[0][1]

    t0k = times[0]
    t_last = times[-1]
    t_src = float(np.clip(t_src, t0k, t_last))

    if mode == "nearest":
        return interp_nearest(keys, times, t_src)

    i = bisect.bisect_right(times, t_src) - 1
    i = max(0, min(i, len(keys) - 2))
    t0, f0 = keys[i]
    t1, f1 = keys[i + 1]
    if t1 <= t0 + 1e-9:
        return f0
    alpha = (t_src - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if mode == "blend":
        return interp_blend_pair(f0, f1, alpha)

    if mode == "flow":
        assert flow_cache is not None
        seg = (i, i + 1)
        if seg not in flow_cache:
            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            flow01 = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            flow10 = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            flow_cache[seg] = (flow01, flow10)
        flow01, flow10 = flow_cache[seg]
        if xy is not None:
            x, y = xy
        else:
            h, w = flow01.shape[:2]
            x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx0 = x - alpha * flow01[:, :, 0]
        my0 = y - alpha * flow01[:, :, 1]
        w0 = cv2.remap(f0, mx0, my0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mx1 = x - (1.0 - alpha) * flow10[:, :, 0]
        my1 = y - (1.0 - alpha) * flow10[:, :, 1]
        w1 = cv2.remap(f1, mx1, my1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(w0, 1.0 - alpha, w1, alpha, 0)

    raise ValueError(f"неизвестный --interp {mode}")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.normpath(os.path.join(here, "..", "data/city", "trm.169.007.avi"))

    p = argparse.ArgumentParser(
        description="Slow motion: выход с постоянным FPS, промежуточные кадры между ключами."
    )
    p.add_argument("-i", "--input", default=default_in, help="Входное видео")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Выходной файл (.mp4). По умолчанию: рядом с входом, суффикс",
    )
    p.add_argument(
        "-f",
        "--factor",
        type=float,
        default=4.0,
        help="Во сколько раз длиннее таймлайн исходника. Должно быть > 1.",
    )
    p.add_argument(
        "--output-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="FPS выходного файла (по умолчанию как у исходника).",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Ключевые кадры с файла с этой частотой по времени исходника.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="N",
        help="Каждый N-й кадр как ключ. Нельзя вместе с --sample-fps.",
    )
    p.add_argument(
        "--interp",
        choices=("blend", "flow", "nearest"),
        default="blend",
        help="Как дорисовывать между ключами: blend | flow | nearest.",
    )
    args = p.parse_args()

    if args.factor <= 1.0:
        print("factor должен быть > 1", file=sys.stderr)
        return 1

    if args.stride is not None and args.stride < 1:
        print("--stride должен быть >= 1", file=sys.stderr)
        return 1

    if args.stride is not None and args.sample_fps is not None:
        print("Задайте только одно: --stride или --sample-fps", file=sys.stderr)
        return 1

    video_path = os.path.abspath(args.input)
    if not os.path.isfile(video_path):
        print(f"Файл не найден: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}", file=sys.stderr)
        return 1

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 1e-3:
        in_fps = 30.0

    out_fps = args.output_fps if args.output_fps is not None else in_fps
    if out_fps <= 1e-6:
        print("--output-fps должен быть > 0", file=sys.stderr)
        cap.release()
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ret, probe = cap.read()
        if not ret:
            print("Пустое видео или не удалось прочитать размер кадра.", file=sys.stderr)
            cap.release()
            return 1
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    keys = collect_keyframes(cap, in_fps, args.sample_fps, args.stride)
    cap.release()

    if not keys:
        print("Нет ключевых кадров.", file=sys.stderr)
        return 1

    times = [t for t, _ in keys]
    duration_src = (n_src / in_fps) if n_src > 0 else times[-1] + 1.0 / in_fps
    duration_src = max(duration_src, times[-1] + 1e-3)

    n_out = max(1, int(round(duration_src * args.factor * out_fps)))
    flow_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = (
        {} if args.interp == "flow" else None
    )
    xy_grid: tuple[np.ndarray, np.ndarray] | None = None
    if args.interp == "flow":
        xy_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        root, ext = os.path.splitext(video_path)
        ext = ext if ext else ".mp4"
        tag = f"_slow{args.factor:g}_{args.interp}"
        if args.stride is not None and args.stride > 1:
            tag += f"_st{args.stride}"
        elif args.sample_fps is not None:
            tag += f"_s{args.sample_fps:g}"
        out_path = f"{root}{tag}{ext}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        writer, used_path = open_writer(out_path, out_fps, (w, h))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    for j in range(n_out):
        t_out = j / out_fps
        t_src = t_out / args.factor
        t_src = min(t_src, times[-1])
        frame = frame_at_time(keys, times, t_src, args.interp, flow_cache, xy_grid)
        writer.write(frame)
        if (j + 1) % 100 == 0:
            print(f"Записано кадров: {j + 1} / {n_out}")

    writer.release()

    eff_desc = "все кадры"
    if args.stride is not None and args.stride > 1:
        eff_desc = f"stride={args.stride}"
    elif args.sample_fps is not None:
        eff_desc = f"sample-fps={args.sample_fps:g}"

    print(
        f"Готово: {n_out} кадров → {used_path}\n"
        f"  исходник ~{duration_src:.4g} с, {in_fps:.3f} fps; ключи: {len(keys)} ({eff_desc}); "
        f"выход {out_fps:.4g} fps, интерполяция={args.interp}, ×{args.factor:g} к длительности"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
