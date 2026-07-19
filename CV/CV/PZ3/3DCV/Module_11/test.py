import cv2

for w in range(5, 11):
    for h in range(4, 9):
        cap = cv2.VideoCapture("./9pz/experiment/xt1.021.003.left.avi")
        for i in range(0, 300, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(gray, (w, h))
            if found:
                print(f"НАЙДЕНА ДОСКА {w}x{h} на кадре {i}!")
        cap.release()