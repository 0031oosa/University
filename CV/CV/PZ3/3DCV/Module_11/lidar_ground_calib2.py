"""
ПЗ №9 — Совместная калибровка LiDAR + камеры
=============================================
Алгоритм:
  1. Читаем grab_msec_timestamps.csv → строим синхронные тройки
     (кадр левой камеры, кадр правой камеры, скан LiDAR)
  2. Для каждой тройки:
       - ищем шахматную доску на обоих кадрах (OpenCV)
       - ищем плоскость доски в облаке точек (RANSAC)
       - решаем PnP → получаем pose доски в системе камеры
  3. Сохраняем результаты в JSON

Запуск:
  python lidar_camera_calib.py

Зависимости:
  pip install opencv-python numpy pandas laspy
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import laspy

# ============================================================
# КОНФИГУРАЦИЯ  — при необходимости поменяй пути
# ============================================================
DATA_DIR       = "./9pz/experiment"
LIDAR_DIR      = os.path.join(DATA_DIR, "robosenseCapture")
LEFT_VIDEO     = os.path.join(DATA_DIR, "xt1.021.003.left.avi")
RIGHT_VIDEO    = os.path.join(DATA_DIR, "xt1.021.003.right.avi")
LEFT_CALIB     = os.path.join(DATA_DIR, "leftImage.yml")
RIGHT_CALIB    = os.path.join(DATA_DIR, "rightImage.yml")
CSV_PATH       = os.path.join(DATA_DIR, "grab_msec_timestamps.csv")
GROUND_JSON    = "./lidar_ground_calib.json"
RESULTS_DIR    = "./results"

# Параметры шахматной доски (внутренние углы)
BOARD_W        = 7
BOARD_H        = 7
SQUARE_SIZE_M  = 1.0   # размер клетки в метрах — уточни по реальной доске!

# Синхронизация: максимальная разница меток времени между сенсорами (мс)
SYNC_TOL_MS    = 60

# Фильтрация LiDAR: ищем доску только в этом диапазоне дистанций и высот
BOARD_X_MIN    =  1.0    # минимум по оси X (вперёд от машины), м
BOARD_X_MAX    = 10.0    # максимум по оси X
BOARD_Z_MIN    =  0.3    # минимум по высоте (отрезаем землю)
BOARD_Z_MAX    =  2.5    # максимум по высоте

# RANSAC для поиска плоскости в LiDAR
RANSAC_THRESH  = 0.025   # м — точка считается inlier если ближе этого порога к плоскости
RANSAC_ITERS   = 500
MIN_INLIERS    = 30      # минимум inlier-точек чтобы считать плоскость найденной


# ============================================================
# 1. КАЛИБРОВКА КАМЕР
# ============================================================
def read_camera_calib(yml_path: str):
    """Читает матрицу K и коэффициенты дисторсии D из OpenCV yml-файла."""
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Не удалось открыть файл калибровки: {yml_path}")
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return K, D


# ============================================================
# 2. СИНХРОНИЗАЦИЯ
# ============================================================
def build_sync_triplets(csv_path: str, lidar_dir: str, tol_ms: int = 60):
    """
    Читает CSV и строит список синхронных троек:
      { left_frame_idx, right_frame_idx, lidar_file, time_ms }

    В CSV каждая строка содержит данные одного сенсора.
    Левая и правая камеры нумеруются порядком появления в CSV.
    LiDAR-файл ищется по timestamp в имени файла.
    """
    df = pd.read_csv(csv_path)

    # Строки с данными каждого сенсора
    left_df  = df[df['left'].notna()].reset_index(drop=True)
    right_df = df[df['right'].notna()].reset_index(drop=True)
    lidar_df = df[df['robosenseCapture'].notna()].reset_index(drop=True)

    # Индекс файлов LiDAR: timestamp → путь к файлу
    lidar_files = {}
    for fname in os.listdir(lidar_dir):
        if fname.endswith('.laz'):
            # Имя вида: xt1.021.003.robosenseCapture_620069.laz
            ts = int(fname.split('_')[-1].replace('.laz', ''))
            lidar_files[ts] = os.path.join(lidar_dir, fname)

    triplets = []
    for l_idx in range(len(left_df)):
        l_time = float(left_df.loc[l_idx, 'left'])

        # Ближайший кадр правой камеры
        r_diffs = (right_df['right'].astype(float) - l_time).abs()
        r_idx   = r_diffs.idxmin()
        r_time  = float(right_df.loc[r_idx, 'right'])
        if abs(r_time - l_time) > tol_ms:
            continue

        # Ближайший скан LiDAR
        lid_diffs = (lidar_df['robosenseCapture'].astype(float) - l_time).abs()
        lid_idx   = lid_diffs.idxmin()
        lid_time  = int(lidar_df.loc[lid_idx, 'robosenseCapture'])
        if abs(lid_time - l_time) > tol_ms:
            continue

        # Ищем файл по timestamp
        lidar_file = lidar_files.get(lid_time)
        if lidar_file is None:
            # Ищем ближайший по ключу
            closest_ts = min(lidar_files.keys(), key=lambda t: abs(t - lid_time))
            if abs(closest_ts - lid_time) < tol_ms:
                lidar_file = lidar_files[closest_ts]
            else:
                continue

        triplets.append({
            'left_frame_idx':  l_idx,
            'right_frame_idx': int(r_idx),
            'lidar_file':      lidar_file,
            'time_ms':         l_time,
        })

    print(f"  Найдено синхронных троек: {len(triplets)}")
    return triplets


# ============================================================
# 3. ЧТЕНИЕ КАДРА ИЗ ВИДЕО
# ============================================================
def read_video_frame(video_path: str, frame_idx: int):
    """Читает кадр из .avi по порядковому индексу."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ============================================================
# 4. ПОИСК ШАХМАТНОЙ ДОСКИ В ОБЛАКЕ ТОЧЕК (RANSAC)
# ============================================================
def find_board_plane_in_lidar(laz_path: str):
    """
    Ищет плоскость шахматной доски в облаке точек методом RANSAC.
    Возвращает (normal, d, inlier_points) или (None, None, None).
    """
    las = laspy.read(laz_path)
    pts = np.vstack([las.x, las.y, las.z]).T  # (N, 3)

    # Пространственная фильтрация: оставляем только зону перед машиной
    mask = (
        (pts[:, 0] > BOARD_X_MIN) & (pts[:, 0] < BOARD_X_MAX) &
        (pts[:, 2] > BOARD_Z_MIN) & (pts[:, 2] < BOARD_Z_MAX)
    )
    pts = pts[mask]

    if len(pts) < MIN_INLIERS:
        return None, None, None

    best_inliers_mask = np.zeros(len(pts), dtype=bool)
    best_normal = None
    best_d = None

    for _ in range(RANSAC_ITERS):
        idx = np.random.choice(len(pts), 3, replace=False)
        p1, p2, p3 = pts[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-6:
            continue
        normal /= n_len
        d = -np.dot(normal, p1)

        dists = np.abs(pts @ normal + d)
        inliers_mask = dists < RANSAC_THRESH

        if inliers_mask.sum() > best_inliers_mask.sum():
            best_inliers_mask = inliers_mask
            best_normal = normal
            best_d = d

    if best_normal is None or best_inliers_mask.sum() < MIN_INLIERS:
        return None, None, None

    return best_normal, best_d, pts[best_inliers_mask]


# ============================================================
# 5. ОСНОВНОЙ ЦИКЛ
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Калибровка камер ---
    print("Читаем калибровку камер...")
    K_left,  D_left  = read_camera_calib(LEFT_CALIB)
    K_right, D_right = read_camera_calib(RIGHT_CALIB)
    print(f"K_left =\n{K_left}")

    # --- Ground calib (из Части 1) ---
    with open(GROUND_JSON) as f:
        ground_calib = json.load(f)
    print(f"Высота LiDAR: {ground_calib['height_m']:.3f} м, "
          f"pitch: {ground_calib['pitch_deg']:.2f}°, "
          f"roll: {ground_calib['roll_deg']:.2f}°")

    # --- Синхронизация ---
    print("\nСтроим синхронные тройки...")
    triplets = build_sync_triplets(CSV_PATH, LIDAR_DIR, SYNC_TOL_MS)
    if not triplets:
        print("Синхронных троек не найдено! Проверь пути и допуск SYNC_TOL_MS.")
        return

    # 3D точки углов доски в её локальной СК
    objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE_M

    board_size = (BOARD_W, BOARD_H)
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    results = []

    cv2.namedWindow("Cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cameras", 1280, 400)

    print(f"\nОбрабатываем {len(triplets)} троек. Нажми Q для выхода.\n")

    for i, triplet in enumerate(triplets):
        print(f"[{i+1}/{len(triplets)}] t={triplet['time_ms']:.0f} мс", end="  ")

        # --- Читаем кадры ---
        frame_l = read_video_frame(LEFT_VIDEO,  triplet['left_frame_idx'])
        frame_r = read_video_frame(RIGHT_VIDEO, triplet['right_frame_idx'])
        if frame_l is None or frame_r is None:
            print("→ ошибка чтения кадра")
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # --- Поиск доски на камерах ---
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, cb_flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, cb_flags)

        # Визуализация
        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        cv2.drawChessboardCorners(vis_l, board_size, corners_l, ret_l)
        cv2.drawChessboardCorners(vis_r, board_size, corners_r, ret_r)

        # Подпись статуса
        status_l = "FOUND" if ret_l else "NOT FOUND"
        status_r = "FOUND" if ret_r else "NOT FOUND"
        cv2.putText(vis_l, f"LEFT  {status_l}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if ret_l else (0,0,255), 2)
        cv2.putText(vis_r, f"RIGHT {status_r}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if ret_r else (0,0,255), 2)

        combined = np.hstack([vis_l, vis_r])
        cv2.imshow("Cameras", combined)
        key = cv2.waitKey(30)
        if key == ord('q'):
            print("\nВыход по нажатию Q")
            break

        if not ret_l or not ret_r:
            print(f"→ доска не найдена (L={ret_l}, R={ret_r})")
            continue

        # Уточняем углы субпиксельно
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), subpix_criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), subpix_criteria)

        # PnP: положение доски в системе левой камеры
        ok, rvec, tvec = cv2.solvePnP(objp, corners_l, K_left, D_left)
        if not ok:
            print("→ solvePnP не сошёлся")
            continue
        R_board_in_cam, _ = cv2.Rodrigues(rvec)

        # --- Поиск доски в LiDAR ---
        normal, d, board_pts = find_board_plane_in_lidar(triplet['lidar_file'])
        if normal is None:
            print(f"→ камера OK, LiDAR: плоскость не найдена")
            continue

        board_center = board_pts.mean(axis=0)
        print(f"→ OK! "
              f"cam_t=[{tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f}] м  "
              f"lidar_center=[{board_center[0]:.2f}, {board_center[1]:.2f}, {board_center[2]:.2f}] м  "
              f"inliers={len(board_pts)}")

        results.append({
            'time_ms':            int(triplet['time_ms']),
            'left_frame_idx':     triplet['left_frame_idx'],
            'right_frame_idx':    triplet['right_frame_idx'],
            'lidar_file':         os.path.basename(triplet['lidar_file']),
            # Положение доски в системе левой камеры
            'rvec_cam':           rvec.flatten().tolist(),
            'tvec_cam':           tvec.flatten().tolist(),
            'R_board_in_cam':     R_board_in_cam.tolist(),
            # Положение доски в системе LiDAR
            'board_normal_lidar': normal.tolist(),
            'board_d_lidar':      float(d),
            'board_center_lidar': board_center.tolist(),
            'num_lidar_inliers':  int(len(board_pts)),
        })

    cv2.destroyAllWindows()

    # --- Сохраняем результаты ---
    out_path = os.path.join(RESULTS_DIR, "lidar_camera_calib.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Обработано кадров с доской: {len(results)} из {len(triplets)}")
    print(f"Результаты сохранены: {out_path}")

    if len(results) > 0:
        print("\nПример первого результата:")
        r = results[0]
        print(f"  Доска в камере: t = {r['tvec_cam']}")
        print(f"  Доска в LiDAR:  center = {r['board_center_lidar']}")
        print(f"  Нормаль плоскости: {r['board_normal_lidar']}")


if __name__ == "__main__":
    main()