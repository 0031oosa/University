import os

path = r'D:\Shared\Documents\Уник\CV\PZ3'

videos = [f for f in os.listdir(path) if f.endswith('.mov')]

print(videos)