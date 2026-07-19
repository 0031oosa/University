import cv2
import numpy as np


def load_relative_param(calib_file:str) ->dict:
    # Загружаем YAML файл с помощью OpenCV
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise IOError(f"Не удалось открыть файл калибровки: {calib_file}")

    # Извлекаем параметры
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    calib = dict()
    calib["R_rel"] = R
    calib["T_rel"] = T
    return calib

def load_images(filepath):
        images = list()
        cap = cv2.VideoCapture(filepath)
        i=0 #переменная отслеживает каждый третий кадр
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                images.append(frame)
            else:
                cap.release()
        return np.array(images)

def load_calibration_params(calib_file):
    """
    Загрузка параметров калибровки стереопары из YAML файла

    Параметры:
    calib_file: путь к YAML файлу с параметрами калибровки

    Возвращает:
    Словарь с параметрами калибровки для левой и правой камер
    """
    # Загружаем YAML файл с помощью OpenCV
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise IOError(f"Не удалось открыть файл калибровки: {calib_file}")

    # Извлекаем параметры
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    r = fs.getNode("r").mat()  # Вектор Родригеса (углы поворота)
    t = fs.getNode("t").mat()  # Вектор переноса
    sz_node = fs.getNode("sz")  # Размер изображения
    image_size = [int(sz_node.at(0).real()), int(sz_node.at(1).real())]

    # Преобразуем вектор Родригеса в матрицу поворота (3x3)
    # Для стереопары у нас есть только одно значение r, предполагаем что это поворот правой камеры относительно левой
    R, _ = cv2.Rodrigues(r)

    # Важно: Для правильной работы stereoRectify, левая камера должна иметь 
    # единичную матрицу поворота и нулевой вектор переноса,
    # а правая камера - матрицу поворота R и вектор переноса T

    calib_params = {
        'camera_matrix': K,
        'dist_coeffs': D,
        'R': R,  # Матрица поворота правой камеры относительно левой
        'T': t,  # Вектор переноса правой камеры относительно левой
        'image_size': image_size,
        'rotation_vector': r,  # Исходный вектор Родригеса
        'translation_vector': t
    }
    fs.release()
    return calib_params