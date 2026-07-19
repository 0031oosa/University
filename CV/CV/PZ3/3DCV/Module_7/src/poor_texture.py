import cv2
import numpy as np


def configure_stereo_for_poor_texture(stereo, texture_level='low'):
    """
    Настройка StereoBM для разных уровней текстуры

    texture_level: 'very_low', 'low', 'medium', 'high'
    """

    if texture_level == 'very_low':
        # Очень плохая текстура (однородное поле, песок)
        stereo.setBlockSize(21)        # Большой блок
        stereo.setNumDisparities(128)   # Средний диапазон
        stereo.setPreFilterSize(15)     # Большой префильтр
        stereo.setPreFilterCap(63)      # Максимальное ограничение
        stereo.setTextureThreshold(5)   # Очень низкий порог
        stereo.setUniquenessRatio(3)    # Низкая уникальность
        stereo.setSpeckleWindowSize(200) # Большое окно для удаления шумов
        stereo.setSpeckleRange(64)       # Большой диапазон

    elif texture_level == 'low':
        # Низкая текстура (трава, поле с редкими растениями)
        stereo.setBlockSize(15)         # Средне-большой блок
        stereo.setNumDisparities(128)    # Средний диапазон
        stereo.setPreFilterSize(9)       # Средний префильтр
        stereo.setPreFilterCap(31)       # Среднее ограничение
        stereo.setTextureThreshold(10)   # Низкий порог
        stereo.setUniquenessRatio(5)     # Средняя уникальность
        stereo.setSpeckleWindowSize(150) # Среднее окно
        stereo.setSpeckleRange(32)       # Средний диапазон

    elif texture_level == 'medium':
        # Средняя текстура (кусты, небольшие растения)
        stereo.setBlockSize(11)          # Средний блок
        stereo.setNumDisparities(128)     # Средний диапазон
        stereo.setPreFilterSize(7)        # Малый префильтр
        stereo.setPreFilterCap(21)        # Малый cap
        stereo.setTextureThreshold(15)    # Средний порог
        stereo.setUniquenessRatio(8)      # Средняя уникальность
        stereo.setSpeckleWindowSize(100)  # Малое окно
        stereo.setSpeckleRange(16)        # Малый диапазон

    elif texture_level == 'high':
        # Высокая текстура (деревья, сложные объекты)
        stereo.setBlockSize(5)            # Малый блок
        stereo.setNumDisparities(112)      # Стандартный диапазон
        stereo.setPreFilterSize(5)         # Малый префильтр
        stereo.setPreFilterCap(15)         # Малый cap
        stereo.setTextureThreshold(20)     # Высокий порог
        stereo.setUniquenessRatio(10)      # Высокая уникальность
        stereo.setSpeckleWindowSize(50)    # Очень малое окно
        stereo.setSpeckleRange(8)          # Очень малый диапазон

    return stereo



def check_rectification_quality(rectifiedL, rectifiedR, num_points=100):
    """
    Проверка качества ректификации путем поиска соответствующих точек

    Параметры:
    rectifiedL, rectifiedR: ректифицированные изображения
    num_points: количество точек для проверки
    """
    # Преобразуем в оттенки серого если нужно
    if len(rectifiedL.shape) == 3:
        grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
    else:
        grayL = rectifiedL
        grayR = rectifiedR

    # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ для goodFeaturesToTrack
    max_corners = 500  # Максимальное количество углов
    quality_level = 0.0001  # Очень низкий порог качества (было 0.01)
    min_distance = 5  # Минимальное расстояние между углами (было 10)
    block_size = 3  # Размер блока для вычисления производных
    use_harris = True  # Использовать детектор Harris
    k = 0.04  # Параметр Harris

    # Находим хорошие точки для отслеживания на левом изображении
    featuresL = cv2.goodFeaturesToTrack(grayL,
        max_corners,
        quality_level,
        min_distance,
        blockSize=block_size,
        useHarrisDetector=use_harris,
        k=k)

    if featuresL is None:
        print("Не удалось найти достаточно точек для проверки")
        return

    # Находим соответствующие точки на правом изображении с помощью оптического потока
    featuresR, status, _ = cv2.calcOpticalFlowPyrLK(grayL, grayR, featuresL, None)

    # Создаем визуализацию
    visl = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
    visr = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)
    vis = np.hstack((visl, visr))

    good_points = 0
    total_y_diff = 0

    for i, (ptL, ptR) in enumerate(zip(featuresL, featuresR)):
        if status[i] == 1:
            xL, yL = ptL.ravel()
            xR, yR = ptR.ravel()

            # Рисуем точки
            cv2.circle(vis, (int(xL), int(yL)), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(xR) + grayL.shape[1], int(yR)), 5, (0, 255, 0), -1)

            # Рисуем линию, соединяющую соответствующие точки
            cv2.line(vis, (int(xL), int(yL)),
                    (int(xR) + grayL.shape[1], int(yR)), (0, 255, 0), 1)

            # Проверяем разницу по y (должна быть близка к 0 при хорошей ректификации)
            y_diff = abs(yL - yR)
            total_y_diff += y_diff
            good_points += 1

    h, w = vis.shape[:2]
    vis_new = cv2.resize(vis, (w//2, h//2))

    if good_points > 0:
        avg_y_diff = total_y_diff / good_points
        print(f"Проверка качества ректификации:")
        print(f"  Найдено соответствующих точек: {good_points}")
        print(f"  Средняя разница по y: {avg_y_diff:.2f} пикселей")

        if avg_y_diff < 1.0:
            print("  ✓ Отличное качество ректификации!")
        elif avg_y_diff < 3.0:
            print("  ✓ Хорошее качество ректификации")
        else:
            print("  ⚠ Среднее качество ректификации")

    # Отображаем результат
    cv2.imshow("Rectification Quality Check", vis_new)
    cv2.waitKey(1)

    return avg_y_diff