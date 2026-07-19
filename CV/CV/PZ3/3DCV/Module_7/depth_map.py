import numpy as np
import cv2
from pathlib import Path
import open3d as o3d


from src.poor_texture import configure_stereo_for_poor_texture
from src.point_cloud import AsyncPointCloudVisualizer
from src.load_calibration_params import load_calibration_params, load_images, load_relative_param


def preprocess_image(image):
    # Произведите необходимую предварительную обработку изображения
    # (например, сглаживание, устранение шума и т. д.)
    # Сглаживание изображения с помощью фильтра Гаусса
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # В этом примере просто преобразуем изображение в оттенки серого
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Улучшение контраста изображения
    image = cv2.equalizeHist(image)
    return image

def visualize_epipolar_lines(imgL, imgR, num_lines=20):
    """
    Функция для визуализации эпиполярных линий на паре изображений
    Параметры:
    imgL, imgR - исходные изображения (левое и правое)
    num_lines - количество отображаемых линий
    """
    # Создаем копии изображений для рисования
    visL = imgL.copy()
    visR = imgR.copy()

    # Если изображения цветные, преобразуем в оттенки серого для лучшей видимости линий
    if len(visL.shape) == 3:
        grayL = cv2.cvtColor(visL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(visR, cv2.COLOR_BGR2GRAY)
        visL = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
        visR = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)

    height = visL.shape[0]

    # Рисуем горизонтальные линии через равные промежутки
    step = height // (num_lines + 1)

    for i in range(1, num_lines + 1):
        y = i * step
        # Выбираем случайный цвет для каждой линии
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Рисуем линии на обоих изображениях
        cv2.line(visL, (0, y), (visL.shape[1] - 1, y), color, 1)
        cv2.line(visR, (0, y), (visR.shape[1] - 1, y), color, 1)
        # Добавляем небольшие маркеры на линиях
        cv2.circle(visL, (visL.shape[1] // 4, y), 3, color, -1)
        cv2.circle(visR, (visR.shape[1] // 4, y), 3, color, -1)
    return visL, visR

def rectify_stereo_images(imgL, imgR, calib_L:dict, calib_R:dict, calib: dict):
    """
    Ректификация стереопары с использованием реальных параметров калибровки

    Параметры:
    imgL, imgR: исходные изображения
    calib_params: словарь с параметрами калибровки стереопары

    Возвращает:
    rectifiedL, rectifiedR: ректифицированные изображения
    Q: матрица репроекции для преобразования disparity в 3D
    """
    h, w = imgL.shape[:2]

    # Извлечение параметров калибровки
    camera_matrix_left = calib_L['camera_matrix']
    dist_coeffs_left = calib_L['dist_coeffs']
    camera_matrix_right = calib_R['camera_matrix']
    dist_coeffs_right = calib_R['dist_coeffs']
    R = calib['R_rel']  # Матрица поворота правой камеры относительно левой
    T = calib['T_rel']  # Вектор переноса правой камеры относительно левой

    # Размер изображения из калибровки
    calib_size = calib_L['image_size']

    print(f"Параметры калибровки для размера {calib_size}")

    # Параметр alpha определяет обрезку черных областей после ректификации
    # alpha=0 - обрезать черные области (максимальное обрезание)
    # alpha=1 - сохранить все пиксели (появляются черные области)
    alpha = 0  # Можно изменить на 1, если нужны все пиксели

    # Вычисление параметров ректификации
    print("Вычисление параметров ректификации...")
    rectL, rectR, proj_matrixL, proj_matrixR, Q, roiL, roiR = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        (w, h),  # размер изображения
        R, T,    # поворот и перенос между камерами
        alpha=alpha,
        newImageSize=(w, h)  # можно указать другой размер для выходных изображений
    )

    print(f"  ROI левой камеры: {roiL}")
    print(f"  ROI правой камеры: {roiR}")

    # Создание карт для ректификации
    print("Создание карт ректификации...")
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, rectL, proj_matrixL,
        (w, h), cv2.CV_32FC1
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, rectR, proj_matrixR,
        (w, h), cv2.CV_32FC1
    )

    # Применение ректификации
    print("Применение ректификации к изображениям...")
    rectifiedL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Опционально: обрезаем до ROI для удаления черных краев
    if alpha == 0 and roiL[2] > 0 and roiL[3] > 0:
        x, y, w_roi, h_roi = roiL
        rectifiedL = rectifiedL[y:y+h_roi, x:x+w_roi]
        rectifiedR = rectifiedR[y:y+h_roi, x:x+w_roi]
        print(f"Изображения обрезаны до ROI: {w_roi}x{h_roi}")

    return rectifiedL, rectifiedR, Q, (roiL, roiR)

def visualize_stereo_pair(imgL, imgR, window_name="Stereo Pair", delay=200):
    """
    Функция для визуализации стереопары рядом
    """
    # Проверяем, что изображения имеют одинаковую высоту
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]

    # Если изображения в оттенках серого, преобразуем в цветные для конкатенации
    if len(imgL.shape) == 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    if len(imgR.shape) == 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # Изменяем размер правого изображения, если высоты не совпадают
    if hL != hR:
        scale = hL / hR
        new_w = int(wR * scale)
        imgR = cv2.resize(imgR, (new_w, hL))

    # Объединяем изображения горизонтально
    combined = np.hstack((imgL, imgR))

    # Добавляем разделительную линию
    cv2.line(combined, (wL, 0), (wL, hL-1), (255, 255, 255), 2)

    # Добавляем подписи
    cv2.putText(combined, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (wL + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображаем
    h, w = combined.shape[:2]
    combined_new = cv2.resize(combined, (w//2, h//2))
    cv2.imshow(window_name, combined_new)
    key = cv2.waitKey(delay)
    return key

def visualize_point_cloud_async(disparity, original_image, Q_matrix=None, max_distance=50.0, voxel_size=0.01):

    points_3D = cv2.reprojectImageTo3D(disparity, Q_matrix)
    depth_map = points_3D[:, :, 2]
    # Создаем маску валидных точек
    valid_mask = (depth_map > 0) & (depth_map < max_distance) & ~np.isnan(depth_map)

    # Извлекаем точки и цвета
    points = points_3D[valid_mask]

    # Получаем цвета из оригинального изображения
    colors = original_image[valid_mask].copy()
    colors = colors / 255.0

    # Создаем облако точек Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Опционально: уменьшаем плотность для производительности
    if voxel_size > 0 and len(points) > 100000:
        print("Выполняется даунсэмплинг...")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"После даунсэмплинга: {len(pcd.points)} точек")

    return pcd


def main(dir: str):
    # Загружаем видео с левой и правой камеры
    pathL = str(Path(dir,'kem.011.001.left.avi'))
    pathR = str(Path(dir,'kem.011.001.right.avi'))
    imagesL = load_images(pathL)
    imagesR = load_images(pathR)
    calib_l = load_calibration_params(str(Path(dir,'calib/cam_plg_left.yml')))
    calib_r = load_calibration_params(str(Path(dir,'calib/cam_plg_righ.yml')))
    calib = load_relative_param(str(Path(dir,'calib/extrinsics.yml')))

    print(f"Загружено {len(imagesL)} кадров")

    # Создание объекта для вычисления карты глубины
    stereo = cv2.StereoBM_create()
    stereo = configure_stereo_for_poor_texture(stereo, 'low')

    point_cloud_visualizer = AsyncPointCloudVisualizer()
    first_frame_processed = True

    # Обработка изображений и вычисление карты глубины
    for idx, (imgL, imgR) in enumerate(zip(imagesL, imagesR)):
        print(f"Обработка кадра {idx + 1}/{len(imagesL)}")

        # визуализация стереопар
        visualize_stereo_pair(imgL, imgR)

        rectifiedL, rectifiedR, Q, rois = rectify_stereo_images(imgL, imgR, calib_l, calib_r, calib)
        # check_rectification_quality(rectifiedL, rectifiedR)

        # ВИЗУАЛИЗАЦИЯ 4: Эпиполярные линии на ректифицированных изображениях
        epi_rectL, epi_rectR = visualize_epipolar_lines(rectifiedL, rectifiedR, num_lines=8)
        visualize_stereo_pair(epi_rectL, epi_rectR, "Epipolar Lines on Rectified Images", 1)

        # Вычисление disparity (на ректифицированных изображениях в оттенках серого)
        gray_rectL = preprocess_image(rectifiedL)
        gray_rectR = preprocess_image(rectifiedR)

        disparity = stereo.compute(gray_rectL, gray_rectR)
        pcd = visualize_point_cloud_async(disparity, imgL, Q)

        if first_frame_processed:
            point_cloud_visualizer.show(pcd)
            first_frame_processed = False
        else:
            point_cloud_visualizer.update(pcd)

        # # ВИЗУАЛИЗАЦИЯ 6: Карта глубины
        depth_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Depth Map", depth_color)

        # Управление воспроизведением
        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):  # Выход
            break
        elif key == ord('p'):  # Пауза
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    point_cloud_visualizer.close()

    pass

if __name__ == '__main__':
    main('/Users/rinat/develop/3DCV/data/stereo/kem.011')
