"""
Двухпроходная стабилизация по оптическому потоку (Lucas–Kanade + affine partial).

1) По всему ролику оцениваются приращения (dx, dy, dtheta) между соседними кадрами.
2) Кумулятивная траектория сглаживается:
   - фильтром Калмана (модель постоянной скорости: положение + скорость, измеряется только положение);
   - или скользящим средним (--smoother ma).
3) На втором проходе каждый кадр вращается/сдвигается на разницу (сглажено − сырое).

Окно: слева исходный кадр, справа стабилизированный. Esc — выход.
"""

import argparse
import sys

import cv2
import numpy as np


def moving_average_1d(curve: np.ndarray, radius: int) -> np.ndarray:
    w = 2 * radius + 1
    kernel = np.ones(w, dtype=np.float64) / w
    pad = np.pad(curve.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def smooth_trajectory(traj: np.ndarray, radius: int) -> np.ndarray:
    out = np.empty_like(traj, dtype=np.float64)
    for j in range(traj.shape[1]):
        out[:, j] = moving_average_1d(traj[:, j], radius)
    return out


def smooth_trajectory_kalman(
    traj: np.ndarray,
    q_vel_xy: float,
    q_vel_theta: float,
    r_xy: float,
    r_theta: float,
) -> np.ndarray:
    """
    Сглаживание кумулятивной траектории (x, y, theta) фильтром Калмана.

    Состояние: [cx, cy, ctheta, vx, vy, vomega]. Переход: постоянная скорость (dt=1).
    Измерение: только [cx, cy, ctheta]. Шум процесса — по скоростям; шум измерений — по трём каналам.
    """
    n = traj.shape[0]
    if n == 0:
        return traj.astype(np.float64)
    if n == 1:
        return traj.astype(np.float64)

    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    qp = 1e-6
    kf.processNoiseCov = np.diag(
        np.array(
            [qp, qp, qp * 0.1, q_vel_xy, q_vel_xy, q_vel_theta],
            dtype=np.float32,
        )
        ** 2
    )
    kf.measurementNoiseCov = np.diag(
        np.array([r_xy, r_xy, r_theta], dtype=np.float32) ** 2
    )
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.5
    kf.statePost = np.zeros((6, 1), dtype=np.float32)
    kf.statePost[:3, 0] = traj[0].astype(np.float32)
    if n > 1:
        kf.statePost[3:, 0] = (traj[1] - traj[0]).astype(np.float32)

    out = np.zeros((n, 3), dtype=np.float64)
    out[0] = traj[0]
    for k in range(1, n):
        kf.predict()
        z = traj[k].reshape(3, 1).astype(np.float32)
        kf.correct(z)
        out[k] = kf.statePost[:3, 0].reshape(-1)
    return out


def affine_incremental(prev_pts: np.ndarray, curr_pts: np.ndarray):
    """Возвращает (dx, dy, da) для преобразования точек с кадра t-1 на кадр t (partial affine)."""
    if prev_pts.shape[0] < 4:
        return 0.0, 0.0, 0.0
    m, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99,
        maxIters=2000,
    )
    if m is None:
        return 0.0, 0.0, 0.0
    dx = float(m[0, 2])
    dy = float(m[1, 2])
    da = float(np.arctan2(m[1, 0], m[0, 0]))
    return dx, dy, da


def warp_stabilize(
    frame_bgr: np.ndarray,
    dx: float,
    dy: float,
    da_rad: float,
    border_scale: float = 1.04,
):
    h, w = frame_bgr.shape[:2]
    center = (w * 0.5, h * 0.5)
    deg = float(np.degrees(da_rad))
    m = cv2.getRotationMatrix2D(center, deg, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    out = cv2.warpAffine(
        frame_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if border_scale != 1.0:
        m2 = cv2.getRotationMatrix2D(center, 0.0, border_scale)
        out = cv2.warpAffine(
            out,
            m2,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


def collect_increments(cap: cv2.VideoCapture, lk_params, feat_params, refresh_every: int):
    ret, prev_bgr = cap.read()
    if not ret:
        return None
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feat_params)
    if prev_pts is None or len(prev_pts) < 4:
        return None

    increments = []
    frame_i = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )

        if curr_pts is None or status is None:
            increments.append((0.0, 0.0, 0.0))
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, **feat_params)
            frame_i += 1
            continue

        ok = status.reshape(-1) == 1
        good_prev = prev_pts[ok].reshape(-1, 2)
        good_curr = curr_pts[ok].reshape(-1, 2)

        if good_prev.shape[0] >= 4:
            increments.append(affine_incremental(good_prev, good_curr))
        else:
            increments.append((0.0, 0.0, 0.0))

        prev_gray = gray
        n_tracked = int(good_curr.shape[0])
        if frame_i % refresh_every == 0 or n_tracked < feat_params["maxCorners"] // 4:
            fresh = cv2.goodFeaturesToTrack(gray, **feat_params)
            if fresh is not None:
                prev_pts = fresh
            elif n_tracked >= 4:
                prev_pts = good_curr.reshape(-1, 1, 2)
            else:
                prev_pts = cv2.goodFeaturesToTrack(gray, **feat_params)
        else:
            prev_pts = good_curr.reshape(-1, 1, 2)

        if prev_pts is None or len(prev_pts) < 8:
            prev_pts = cv2.goodFeaturesToTrack(gray, **feat_params)
        frame_i += 1

    return np.array(increments, dtype=np.float64)


def run(
    video_path: str,
    smoother: str,
    smooth_radius: int,
    max_display_width: int,
    border_scale: float,
    kalman_q_vel_xy: float,
    kalman_q_vel_theta: float,
    kalman_r_xy: float,
    kalman_r_theta: float,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}", file=sys.stderr)
        return 1

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    feat_params = dict(
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=5,
    )

    inc = collect_increments(cap, lk_params, feat_params, refresh_every=15)
    cap.release()

    if inc is None or len(inc) == 0:
        print("Не удалось оценить движение по первым кадрам.", file=sys.stderr)
        return 1

    n_frames = int(inc.shape[0] + 1)
    traj = np.zeros((n_frames, 3), dtype=np.float64)
    for i in range(1, n_frames):
        traj[i] = traj[i - 1] + inc[i - 1]

    if smoother == "kalman":
        sm = smooth_trajectory_kalman(
            traj,
            q_vel_xy=kalman_q_vel_xy,
            q_vel_theta=kalman_q_vel_theta,
            r_xy=kalman_r_xy,
            r_theta=kalman_r_theta,
        )
    else:
        sm = smooth_trajectory(traj, smooth_radius)
    delta = sm - traj

    cap = cv2.VideoCapture(video_path)
    fi = 0
    tag = "Kalman" if smoother == "kalman" else "MA"
    win = f"Original | Stabilized ({tag})  [Esc = quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if fi >= delta.shape[0]:
            break
        d = delta[fi]
        stab = warp_stabilize(bgr, float(d[0]), float(d[1]), float(d[2]), border_scale)
        pair = np.hstack([bgr, stab])
        mw = max_display_width
        if mw > 0 and pair.shape[1] > mw:
            r = mw / pair.shape[1]
            pair = cv2.resize(pair, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, pair)
        fi += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main():
    p = argparse.ArgumentParser(description="Стабилизация по оптическому потоку, превью до/после.")
    p.add_argument(
        "video",
        nargs="?",
        default="../data/opticla_flow.avi",
        help="Путь к видео",
    )
    p.add_argument(
        "--smoother",
        choices=("kalman", "ma"),
        default="kalman",
        help="Сглаживание траектории: Kalman (по умол.) или скользящее среднее (ma)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Только для --smoother ma: радиус сглаживания (кадры)",
    )
    p.add_argument(
        "--kalman-q-vel-xy",
        type=float,
        default=2e-2,
        help="Дисп. шума процесса по vx, vy (больше — фильтр быстрее следует за измерениями)",
    )
    p.add_argument(
        "--kalman-q-vel-theta",
        type=float,
        default=5e-5,
        help="Дисп. шума процесса по угловой скорости (рад/кадр, в квадрате задаётся порядок)",
    )
    p.add_argument(
        "--kalman-r-xy",
        type=float,
        default=4.0,
        help="СКО шума измерения по x, y (пиксели)",
    )
    p.add_argument(
        "--kalman-r-theta",
        type=float,
        default=0.03,
        help="СКО шума измерения по theta (радианы)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1400,
        help="Макс. ширина окна (0 = без уменьшения)",
    )
    p.add_argument(
        "--border-scale",
        type=float,
        default=1.0,
        help="Лёгкий зум после стабилизации, чтобы убрать чёрные поля (1.0 = выкл.)",
    )
    args = p.parse_args()
    return run(
        args.video,
        args.smoother,
        max(1, args.smooth),
        max(0, args.max_width),
        args.border_scale,
        max(1e-12, args.kalman_q_vel_xy),
        max(1e-12, args.kalman_q_vel_theta),
        max(1e-6, args.kalman_r_xy),
        max(1e-9, args.kalman_r_theta),
    )


if __name__ == "__main__":
    raise SystemExit(main())
