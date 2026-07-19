import os
import cv2
import pandas as pd


if __name__ == "__main__":

    files = [file for file in os.listdir(".") if file.find(".mp4") > 0]
    df = pd.read_csv("frame_timestamps.txt")
    tic = df["timestamp[nanosec]"]
    [r,] = tic.shape
    for file in files:
        print(file)
        path = os.path.abspath(os.path.join(".", file))
        cap = cv2.VideoCapture(path)
        flag = True
        count = 0
        while flag:
            f, image = cap.read()
            if flag is False:
                break
            number = tic.iloc[count]
            im_name = os.path.abspath(f"./dataset/cam0/{number}.png")
            cv2.imshow("Window", image)
            if cv2.waitKey(2) == ord("q"):
                break
            cv2.imwrite(im_name, image)
            count = count + 1
            if count == r:
                break
    cv2.destroyAllWindows()