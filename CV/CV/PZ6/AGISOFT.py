import os
import cv2

if __name__ == "__main__":
    dir = "D:\Shared\Documents\Уник\CV\PZ6"
    FRAME_STEP = 6

    images_dir = os.path.join(dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    files = [f for f in os.listdir(dir)
             if f.lower().endswith(".mov")]

    count = 1
    for file in files:
        path = os.path.abspath(os.path.join(dir, file))
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        print(f"Обрабатываю: {file}")

        while True:
            success, im = cap.read()
            if not success:
                break
            if frame_idx % FRAME_STEP == 0:
                im_name = os.path.join(images_dir, f"{count:05d}.png")
                _, buf = cv2.imencode('.png', im)
                with open(im_name, 'wb') as f:
                    f.write(buf.tobytes())
                count += 1
            frame_idx += 1  # ← здесь, снаружи if

        print(f"  → сохранено кадров: {count - 1}")
        cap.release()