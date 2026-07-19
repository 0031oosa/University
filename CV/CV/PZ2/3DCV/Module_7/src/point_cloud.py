import threading
import time
from queue import Queue
import open3d as o3d
import numpy as np


class AsyncPointCloudVisualizer:
    """
    Класс для асинхронной визуализации облака точек с возможностью обновления
    """
    def __init__(self):
        self.visualizer = None
        self.point_cloud = None
        self.is_running = False
        self.thread = None
        self.update_queue = Queue()  # Очередь для новых облаков точек
        self.current_point_cloud = None


    def _run_visualization(self):
        """Запуск визуализатора в отдельном потоке"""
        # Создаем визуализатор
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name="3D Point Cloud (Live Update)",
                                      width=1024, height=768)

        # Создаем начальное облако точек (пустое или из self.point_cloud)
        if self.point_cloud is not None:
            self.current_point_cloud = self.point_cloud
            self.visualizer.add_geometry(self.current_point_cloud)
        else:
            # Создаем пустое облако точек
            self.current_point_cloud = o3d.geometry.PointCloud()
            self.visualizer.add_geometry(self.current_point_cloud)

        # Настройки
        opt = self.visualizer.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 1.0

        # Настройка камеры
        ctr = self.visualizer.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.8)

        self.is_running = True
        update_counter = 0

        print("Визуализатор запущен в отдельном потоке")

        # Цикл визуализации
        while self.is_running:
            # Проверяем, есть ли новые данные в очереди
            if not self.update_queue.empty():
                new_point_cloud = self.update_queue.get_nowait()

                # Обновляем облако точек
                self.current_point_cloud.points = new_point_cloud.points
                self.current_point_cloud.colors = new_point_cloud.colors

                # Важно: сообщаем визуализатору об обновлении геометрии
                self.visualizer.update_geometry(self.current_point_cloud)

                update_counter += 1
                print(f"Облако точек обновлено #{update_counter}")

            # Обновляем рендеринг
            self.visualizer.poll_events()
            self.visualizer.update_renderer()

            time.sleep(0.01)  # Небольшая задержка

        print("Визуализатор завершает работу")
        self.visualizer.destroy_window()

    def show(self, point_cloud):
        """Показать начальное облако точек асинхронно"""
        self.point_cloud = point_cloud

        if self.thread is not None and self.thread.is_alive():
            # Если поток уже запущен, просто обновляем через очередь
            self.update(point_cloud)
            return

        # Запускаем новый поток
        self.thread = threading.Thread(target=self._run_visualization)
        self.thread.daemon = True
        self.thread.start()

    def update(self, point_cloud):
        """Обновить облако точек"""
        if self.is_running:
            self.update_queue.put(point_cloud)
            return True
        return False

    def close(self):
        """Закрыть визуализатор"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("Визуализатор закрыт")