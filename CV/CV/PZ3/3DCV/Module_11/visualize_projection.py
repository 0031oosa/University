"""
PZ9 - LiDAR -> Camera projection visualizer (autoplay)

Controls:
  SPACE - pause/resume
  Q     - quit
  LEFT/RIGHT arrows - prev/next frame manually

Run:
  python visualize_projection.py
"""

import json
import os
import cv2
import numpy as np
import laspy

# ============================================================
DATA_DIR       = "./9pz/experiment"
LIDAR_DIR      = os.path.join(DATA_DIR, "robosenseCapture")
LEFT_VIDEO     = os.path.join(DATA_DIR, "xt1.021.003.left.avi")
LEFT_CALIB     = os.path.join(DATA_DIR, "leftImage.yml")
CALIB_JSON     = "./results/lidar_camera_calib.json"
EXTRINSIC_JSON = "./results/extrinsic_lidar_to_cam.json"

MAX_RANGE_M  = 30.0
MIN_RANGE_M  =  0.5
COLOR_NEAR   =  1.0   # ближняя граница colormap (м) — красный
COLOR_FAR    = 15.0   # дальняя граница colormap (м) — синий
POINT_R      =  3     # радиус точки в пикселях
SHOW_BOARD   = True   # рисовать крест в проекции центра доски из LiDAR
DELAY_MS     = 80     # задержка между кадрами (мс)

# ============================================================
def read_camera_calib(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return K, D

def project_points(pts, R, t, K):
    """Проецирует 3D точки LiDAR на изображение камеры.
    Возвращает: u, v (пиксели), mask_front, depths (camera Z, м)."""
    pts_cam = (R @ pts.T).T + t
    mask_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask_front]

    if len(pts_cam) == 0:
        return np.array([]), np.array([]), mask_front, np.array([])

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    Z = pts_cam[:, 2]
    u = (fx * pts_cam[:,0] / Z + cx).astype(int)
    v = (fy * pts_cam[:,1] / Z + cy).astype(int)
    return u, v, mask_front, Z

def colorize_depth(depths, d_min=0.5, d_max=20.0):
    """Раскрашивает точки по глубине через colormap."""
    norm = np.clip((depths - d_min) / (d_max - d_min), 0, 1)
    indices = (norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(indices.reshape(-1,1), cv2.COLORMAP_JET)
    return colors.reshape(-1, 3)

# ============================================================
print("Loading calibration...")
K, D = read_camera_calib(LEFT_CALIB)

with open(EXTRINSIC_JSON) as f:
    ext = json.load(f)
R = np.array(ext['R_lidar_to_cam'])
t = np.array(ext['t_lidar_to_cam'])

with open(CALIB_JSON) as f:
    frames_data = json.load(f)

print(f"Frames: {len(frames_data)}")

cap = cv2.VideoCapture(LEFT_VIDEO)
h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cv2.namedWindow("LiDAR -> Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LiDAR -> Camera", min(w_img, 1280), min(h_img, 540))

i = 0
paused = False

while i < len(frames_data):
    entry = frames_data[i]
    frame_idx  = entry['left_frame_idx']
    lidar_file = os.path.join(LIDAR_DIR, entry['lidar_file'])

    # Читаем кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or not os.path.exists(lidar_file):
        i += 1
        continue

    # Читаем всё облако точек
    las  = laspy.read(lidar_file)
    pts  = np.vstack([las.x, las.y, las.z]).T

    # Фильтр по дальности
    rng  = np.linalg.norm(pts, axis=1)
    mask = (rng > MIN_RANGE_M) & (rng < MAX_RANGE_M)
    pts  = pts[mask]

    vis = frame.copy()

    if len(pts) > 0:
        u, v, mf, depths = project_points(pts, R, t, K)

        if len(u) > 0:
            colors = colorize_depth(depths, d_min=COLOR_NEAR, d_max=COLOR_FAR)

            # Рисуем в порядке от дальнего к ближнему
            order = np.argsort(-depths)
            h, w  = vis.shape[:2]
            for idx in order:
                pu, pv = int(u[idx]), int(v[idx])
                if 0 <= pu < w and 0 <= pv < h:
                    cv2.circle(vis, (pu, pv), POINT_R,
                               (int(colors[idx,0]), int(colors[idx,1]), int(colors[idx,2])), -1)

    if SHOW_BOARD and 'board_center_lidar' in entry:
        h, w = vis.shape[:2]

        # Зелёный крест: центр доски из LiDAR, спроецированный нашим экстринсиком
        bc_lidar = np.array(entry['board_center_lidar'])
        bc_cam   = R @ bc_lidar + t
        if bc_cam[2] > 0.1:
            bu = int(K[0,0] * bc_cam[0] / bc_cam[2] + K[0,2])
            bv = int(K[1,1] * bc_cam[1] / bc_cam[2] + K[1,2])
            if 0 <= bu < w and 0 <= bv < h:
                cv2.drawMarker(vis, (bu, bv), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
                cv2.circle(vis, (bu, bv), 20, (0, 255, 0), 2)
                cv2.putText(vis, "LiDAR", (bu + 22, bv - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Красный крест: центр доски из камеры (solvePnP rvec/tvec)
        if 'rvec_cam' in entry and 'tvec_cam' in entry:
            import cv2 as _cv2
            rvec_e = np.array(entry['rvec_cam'])
            tvec_e = np.array(entry['tvec_cam'])
            R_b, _ = _cv2.Rodrigues(rvec_e)
            cx_b, cy_b = (7 - 1) / 2.0, (7 - 1) / 2.0   # центр 7x7 доски в клетках
            # Масштаб S уже известен из extrinsic JSON
            S_vis = ext.get('estimated_square_size_m', 0.041)
            c_sq  = R_b @ np.array([cx_b, cy_b, 0.0]) + tvec_e
            c_m   = c_sq * S_vis
            if c_m[2] > 0.1:
                ru = int(K[0,0] * c_m[0] / c_m[2] + K[0,2])
                rv = int(K[1,1] * c_m[1] / c_m[2] + K[1,2])
                if 0 <= ru < w and 0 <= rv < h:
                    cv2.drawMarker(vis, (ru, rv), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
                    cv2.circle(vis, (ru, rv), 20, (0, 0, 255), 2)
                    cv2.putText(vis, "Cam", (ru + 22, rv + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    pts_shown = int(len(u)) if len(pts) > 0 and len(u) > 0 else 0
    cv2.putText(vis, f"Frame {frame_idx}  |  LiDAR pts: {pts_shown}  |  [{i+1}/{len(frames_data)}]",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(vis, "SPACE=pause  Q=quit  <-/-> =prev/next",
                (15, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

    cv2.imshow("LiDAR -> Camera", vis)

    delay = 0 if paused else DELAY_MS
    key   = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == 83 or key == ord('d'):   # стрелка вправо
        i += 1
    elif key == 81 or key == ord('a'):   # стрелка влево
        i = max(0, i - 1)
    elif not paused:
        i += 1

cap.release()
cv2.destroyAllWindows()
print("Done!")