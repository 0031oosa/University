"""
Пример: эвристика «это дорога / земля» по оптическому потоку при движении камеры по городу.

Идея (кратко): при преимущественно поступательном движении вперёд изображение расширяется
от точки схода (focus of expansion, FOE) около горизонта. Точки на плоскости земли в нижней
части кадра дают поток, направленный примерно вдоль луча (пиксель − FOE). Здания и небо
часто нарушают эту модель (другая глубина, сдвиги от поворота).

Ограничения: сильный поворот камеры, боковой дрейф, резкие неровности — модель слабеет;
пороги подбираются под конкретное видео.

Запуск: все поддерживаемые ролики из каталога проигрываются по очереди.
  python ground_from_flow_example.py ../data/city
  python ground_from_flow_example.py ../data/city --pitch-deg 0   # камера без подъёма
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def list_videos(directory: Path):
    if not directory.is_dir():
        return []
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def estimate_foe_from_flow(flow, mag_thresh=1.0, grid_step=12, foe_y_frac=0.4):
    """
    Грубая оценка FOE: для каждой точки сетки продолжаем отрезок [-v] и ищем медиану пересечений.
    foe_y_frac — высота горизонтальной линии пересечения (доля высоты кадра). При наклоне камеры
    вверх горизонт визуально опускается — берите foe_y_frac поближе к 0.48–0.52.
    """
    h, w = flow.shape[:2]
    y_idx, x_idx = np.mgrid[grid_step // 2 : h : grid_step, grid_step // 2 : w : grid_step]
    y_idx = y_idx.astype(np.float32)
    x_idx = x_idx.astype(np.float32)
    fx = flow[y_idx.astype(int), x_idx.astype(int), 0]
    fy = flow[y_idx.astype(int), x_idx.astype(int), 1]
    mag = np.hypot(fx, fy)
    ok = mag > mag_thresh
    if not np.any(ok):
        return np.array([w * 0.5, h * 0.35], dtype=np.float32)

    fx, fy = fx[ok], fy[ok]
    x_idx, y_idx = x_idx[ok], y_idx[ok]
    y_line = h * float(np.clip(foe_y_frac, 0.22, 0.72))
    t = (y_line - y_idx) / (-fy + 1e-6)
    x_inter = x_idx + t * (-fx)
    valid = (t > 0) & (x_inter > -w) & (x_inter < 2 * w)
    if not np.any(valid):
        return np.array([w * 0.5, h * 0.35], dtype=np.float32)
    foe_x = float(np.median(x_inter[valid]))
    foe_y = y_line
    return np.array([foe_x, foe_y], dtype=np.float32)


def ground_likelihood_mask(
    flow,
    foe,
    lower_frac=0.58,
    angle_tol_deg=42,
    mag_min=0.16,
    mag_percentile=8,
):
    """
    Маска «похоже на землю»: нижняя (и средняя) часть кадра + согласованность направления с FOE.

    Порог |v| — перцентиль по всей геометрической полосе roi (а не только по самому низу кадра),
    иначе дальняя дорога с малым потоком отбрасывается.
    """
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ray = np.stack([xx - foe[0], yy - foe[1]], axis=-1)
    ray_len = np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-6
    ray_u = ray / ray_len

    f = flow.astype(np.float32)
    f_len = np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6
    f_u = f / f_len
    cos_align = np.sum(ray_u * f_u, axis=-1)
    mag = np.linalg.norm(f, axis=-1)

    y0 = int(h * (1.0 - lower_frac))
    roi = np.zeros((h, w), dtype=bool)
    roi[y0:, :] = True

    thr = max(mag_min, float(np.percentile(mag[roi], mag_percentile)))

    align_ok = cos_align > np.cos(np.radians(angle_tol_deg))
    ground = roi & align_ok & (mag > thr)
    return ground, thr


def params_for_camera_pitch_deg(pitch_deg: float):
    """
    Подстройка под наклон камеры вверх (pitch_deg > 0): дорога и сходка выше в кадре, |v| вдали меньше.
    При 0° даёт параметры, близкие к исходным «ровным» настройкам.
    """
    t = float(np.clip(pitch_deg / 10.0, 0.0, 2.5))
    return {
        "lower_frac": 0.42 + 0.16 * t,
        "angle_tol_deg": 32.0 + 10.0 * t,
        "mag_min": max(0.08, 0.35 - 0.18 * t),
        "mag_percentile": max(4.0, 15.0 - 7.0 * t),
        "foe_y_frac": 0.40 + 0.08 * t,
    }


def process_video_file(path: Path, pitch_deg: float = 10.0):
    """
    Обрабатывает один файл. Возвращает False, если пользователь нажал Esc (выход из всего).
    Возвращает True, если ролик доиграл или файл не открылся — можно переходить к следующему.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Не удалось открыть: {path}")
        return True

    ret, im = cap.read()
    if not ret:
        print(f"Пустой или битый файл: {path}")
        cap.release()
        return True

    prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    stem = path.name
    gp = params_for_camera_pitch_deg(pitch_deg)

    while True:
        ret, im = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        prev_gray = gray

        foe = estimate_foe_from_flow(flow, foe_y_frac=gp["foe_y_frac"])
        mask, thr = ground_likelihood_mask(
            flow,
            foe,
            lower_frac=gp["lower_frac"],
            angle_tol_deg=gp["angle_tol_deg"],
            mag_min=gp["mag_min"],
            mag_percentile=gp["mag_percentile"],
        )

        vis = im.copy()
        overlay = np.zeros_like(vis)
        overlay[mask] = (0, 200, 0)
        cv2.addWeighted(vis, 0.85, overlay, 0.15, 0, vis)

        cv2.circle(vis, (int(foe[0]), int(foe[1])), 8, (0, 165, 255), 2)
        cv2.putText(
            vis,
            "FOE (estimate)",
            (int(foe[0]) + 10, int(foe[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "Green: lower ROI + flow aligned with expansion from FOE (road/ground)",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"mag > {thr:.2f} (adapt.)",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            stem[:80],
            (10, vis.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Ground from optical flow", vis)
        hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("Flow HSV", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cap.release()
            return False

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ground-from-flow demo: all videos in a folder, played in sequence."
    )
    parser.add_argument(
        "video_dir",
        nargs="?",
        default="../data/city",
        help="Каталог с видео (.avi, .mp4, …). По умолчанию: ../data/city",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=10.0,
        help="Наклон камеры вверх, градусы (0 — как ровный горизонт; ~10 — типичный подъём).",
    )
    args = parser.parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    files = list_videos(video_dir)

    if not files:
        print(
            f"В каталоге нет поддерживаемых видео: {video_dir}\n"
            f"Расширения: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
        return

    print(
        f"Каталог: {video_dir} ({len(files)} файл(ов)), pitch_up={args.pitch_deg:g} deg"
    )
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        if not process_video_file(path, pitch_deg=args.pitch_deg):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
