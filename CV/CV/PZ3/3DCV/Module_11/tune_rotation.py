"""
PZ9 - Interactive rotation tuning
Adjust roll/pitch/yaw correction until LiDAR scan lines look horizontal.

Controls:
  Trackbars: roll, pitch, yaw correction (-180 to +180 deg)
  Q - quit and save corrected extrinsic
"""

import json
import os
import cv2
import numpy as np
import laspy

DATA_DIR       = "./9pz/experiment"
LIDAR_DIR      = os.path.join(DATA_DIR, "robosenseCapture")
LEFT_VIDEO     = os.path.join(DATA_DIR, "xt1.021.003.left.avi")
LEFT_CALIB     = os.path.join(DATA_DIR, "leftImage.yml")
CALIB_JSON     = "./results/lidar_camera_calib.json"
EXTRINSIC_JSON = "./results/extrinsic_lidar_to_cam.json"
OUTPUT_JSON    = "./results/extrinsic_tuned.json"

FRAME_IDX = 50   # фиксированный кадр для настройки

# ============================================================
def read_camera_calib(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return K, D

def Rx(a): 
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def project_and_draw(frame, pts, R, t, K):
    vis = frame.copy()
    pts_cam = (R @ pts.T).T + t
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return vis
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    Z = pts_cam[:,2]
    u = (fx * pts_cam[:,0] / Z + cx).astype(int)
    v = (fy * pts_cam[:,1] / Z + cy).astype(int)
    h, w = vis.shape[:2]
    # colorize by depth
    d_norm = np.clip((Z - 0.5) / 19.5, 0, 1)
    idx = (d_norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(idx.reshape(-1,1), cv2.COLORMAP_JET).reshape(-1,3)
    order = np.argsort(-Z)
    for i in order:
        pu, pv = int(u[i]), int(v[i])
        if 0 <= pu < w and 0 <= pv < h:
            cv2.circle(vis, (pu, pv), 2, (int(colors[i,0]), int(colors[i,1]), int(colors[i,2])), -1)
    return vis

# ============================================================
print("Loading...")
K, D = read_camera_calib(LEFT_CALIB)

with open(EXTRINSIC_JSON) as f:
    ext = json.load(f)
R_base = np.array(ext['R_lidar_to_cam'])
t      = np.array(ext['t_lidar_to_cam'])

with open(CALIB_JSON) as f:
    frames_data = json.load(f)

# Load a fixed frame
entry      = frames_data[FRAME_IDX]
frame_idx  = entry['left_frame_idx']
lidar_file = os.path.join(LIDAR_DIR, entry['lidar_file'])

cap = cv2.VideoCapture(LEFT_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
_, frame = cap.read()
cap.release()

las = laspy.read(lidar_file)
pts = np.vstack([las.x, las.y, las.z]).T
rng = np.linalg.norm(pts, axis=1)
pts = pts[(rng > 0.3) & (rng < 20)]

# ============================================================
cv2.namedWindow("Tune Rotation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tune Rotation", 1280, 540)

# Trackbars: offset +180 so range is 0-360, center=180 means 0 deg correction
cv2.createTrackbar("Roll  (-180..+180)", "Tune Rotation", 180, 360, lambda x: None)
cv2.createTrackbar("Pitch (-180..+180)", "Tune Rotation", 180, 360, lambda x: None)
cv2.createTrackbar("Yaw   (-180..+180)", "Tune Rotation", 180, 360, lambda x: None)

print("Adjust sliders to align LiDAR points with the scene.")
print("Press Q to save and quit.")

while True:
    roll_deg  = cv2.getTrackbarPos("Roll  (-180..+180)", "Tune Rotation") - 180
    pitch_deg = cv2.getTrackbarPos("Pitch (-180..+180)", "Tune Rotation") - 180
    yaw_deg   = cv2.getTrackbarPos("Yaw   (-180..+180)", "Tune Rotation") - 180

    roll  = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw   = np.radians(yaw_deg)

    # Apply correction to base rotation
    R_corr = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = R_corr @ R_base

    vis = project_and_draw(frame, pts, R, t, K)
    cv2.putText(vis, f"Roll:{roll_deg:+.0f}  Pitch:{pitch_deg:+.0f}  Yaw:{yaw_deg:+.0f} deg",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(vis, "Q = save and quit",
                (15, vis.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    cv2.imshow("Tune Rotation", vis)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save tuned extrinsic
R_corr = Rz(np.radians(yaw_deg)) @ Ry(np.radians(pitch_deg)) @ Rx(np.radians(roll_deg))
R_final = R_corr @ R_base
RT = np.eye(4)
RT[:3,:3] = R_final
RT[:3, 3] = t

result = ext.copy()
result['R_lidar_to_cam'] = R_final.tolist()
result['RT_4x4'] = RT.tolist()
result['correction_deg'] = {'roll': roll_deg, 'pitch': pitch_deg, 'yaw': yaw_deg}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Saved tuned extrinsic: {OUTPUT_JSON}")
print(f"Correction applied: roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg} deg")
