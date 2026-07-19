"""
PZ9 - Extrinsic calibration LiDAR -> Camera
============================================
Approach:
  1. R_nominal from coordinate-system analysis (cam_to_vr geometric prior):
       LiDAR (x-fwd, y-left, z-up)  ->  VR (x-right, y-fwd, z-up)  ->  Camera (x-right, y-down, z-fwd)
       R_nominal = cam_to_vr @ R_lidar_to_vr = [[0,-1,0],[0,0,-1],[1,0,0]]

  2. The board is stationary, so all LiDAR normals point in one direction (≈[1,0,0]).
     SVD on normals alone would only constrain 1 axis (the board's facing direction).
     Fix: find the MINIMAL rotation Q that maps R_nominal's predicted normal to the
     observed mean camera normal, then apply Q to R_nominal:
       Q: R_nominal @ n_lidar_mean  ->  n_cam_mean   (corrects sensor mounting offset)
       R = Q @ R_nominal   (preserves the other 2 DOF from R_nominal)

  3. Board center in camera frame uses the full board pose:
       c_cam = (R_board @ [cx,cy,0] + tvec) * S    (center = (3,3) for 7x7 board)

  4. Scale S estimated as median(dist_lidar / dist_cam_sq) across all frames.

  5. Translation estimated as median(c_cam_m - R @ c_lidar) across all frames.

Run: python compute_extrinsic.py
"""

import json
import numpy as np
import cv2

RESULTS_JSON = "./results/lidar_camera_calib.json"
OUTPUT_JSON  = "./results/extrinsic_lidar_to_cam.json"
BOARD_INNER  = (7, 7)   # inner corners (indices 0..6 in each axis)

# ============================================================
with open(RESULTS_JSON) as f:
    data = json.load(f)
print(f"Frames loaded: {len(data)}")

# ============================================================
# Extract per-frame data
# ============================================================
normals_cam    = []
normals_lidar  = []
centers_lidar  = []
centers_cam_sq = []   # board centre in camera frame (square units)
R_boards       = []

cx = (BOARD_INNER[0] - 1) / 2.0   # 3.0  – board centre offset in squares
cy = (BOARD_INNER[1] - 1) / 2.0   # 3.0

for entry in data:
    rvec  = np.array(entry['rvec_cam'])
    tvec  = np.array(entry['tvec_cam'])
    R_board, _ = cv2.Rodrigues(rvec)

    # Board normal in camera frame (3rd column of board rotation = board z-axis)
    n_cam = R_board[:, 2].copy()
    if n_cam[2] < 0:       # ensure it points forward (+z) in camera frame
        n_cam = -n_cam

    # Board normal in LiDAR frame
    n_lid = np.array(entry['board_normal_lidar'])
    if n_lid[0] < 0:       # ensure it points forward (+x) in LiDAR frame
        n_lid = -n_lid

    # Board centre in camera frame (in square units):
    #   corner-0 is at tvec; centre is offset by (cx,cy,0) in board local frame
    c_sq = R_board @ np.array([cx, cy, 0.0]) + tvec

    normals_cam.append(n_cam)
    normals_lidar.append(n_lid)
    centers_lidar.append(np.array(entry['board_center_lidar']))
    centers_cam_sq.append(c_sq)
    R_boards.append(R_board)

normals_cam    = np.array(normals_cam)
normals_lidar  = np.array(normals_lidar)
centers_lidar  = np.array(centers_lidar)
centers_cam_sq = np.array(centers_cam_sq)

# ============================================================
# Step 1: Nominal R from coordinate-system analysis
# ============================================================
# cam_to_vr maps VR vectors -> camera vectors  (from calib.py)
cam_to_vr = np.array([[1,  0,  0],
                       [0,  0, -1],
                       [0,  1,  0]], dtype=float)

# LiDAR->VR: x_lid(fwd)->y_vr(fwd),  y_lid(left)->-x_vr,  z_lid(up)->z_vr
R_lidar_to_vr = np.array([[0, -1, 0],
                           [1,  0, 0],
                           [0,  0, 1]], dtype=float)

R_nominal = cam_to_vr @ R_lidar_to_vr   # = [[0,-1,0],[0,0,-1],[1,0,0]]

print("\n=== Geometric prior (coordinate systems) ===")
print(f"R_nominal:\n{R_nominal}")
print(f"det = {np.linalg.det(R_nominal):.3f}")
print(f"  LiDAR X(fwd)  -> cam {R_nominal @ [1,0,0]}  (should be [0,0,1]  cam-Z fwd)")
print(f"  LiDAR Y(left) -> cam {R_nominal @ [0,1,0]}  (should be [-1,0,0] cam-X left)")
print(f"  LiDAR Z(up)   -> cam {R_nominal @ [0,0,1]}  (should be [0,-1,0] cam-Y up)")

# ============================================================
# Step 2: Correct R using mean board-normal constraint
#
# The board is stationary  =>  n_lidar ≈ const ≈ [1,0,0] for all frames.
# SVD on normals can only constrain the 1st column of R (board facing direction).
# The other 2 DOF come from R_nominal (geometric prior).
#
# Method: find the MINIMAL rotation Q such that
#   Q @ (R_nominal @ n_lidar_mean) = n_cam_mean
# then  R = Q @ R_nominal.
# This preserves the structure of R_nominal for the unconstrained axes.
# ============================================================
n_lidar_mean = np.mean(normals_lidar, axis=0)
n_lidar_mean /= np.linalg.norm(n_lidar_mean)

n_cam_mean = np.mean(normals_cam, axis=0)
n_cam_mean /= np.linalg.norm(n_cam_mean)

# What R_nominal predicts the board normal should be in camera frame
predicted = R_nominal @ n_lidar_mean   # ≈ [0, 0, 1]

# Rodrigues rotation Q: rotate 'predicted' onto 'n_cam_mean'
v1   = predicted / np.linalg.norm(predicted)
v2   = n_cam_mean
axis = np.cross(v1, v2)
sin_a = np.linalg.norm(axis)
cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))

if sin_a < 1e-8:
    Q = np.eye(3)
else:
    axis /= sin_a
    K = np.array([[       0, -axis[2],  axis[1]],
                  [ axis[2],        0, -axis[0]],
                  [-axis[1],  axis[0],        0]])
    Q = cos_a * np.eye(3) + sin_a * K + (1.0 - cos_a) * np.outer(axis, axis)

R = Q @ R_nominal

# Enforce exact rotation matrix via SVD projection
U, _, Vt = np.linalg.svd(R)
R = U @ Vt
if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt

correction_deg = np.degrees(np.arccos(cos_a))
print(f"\n=== Data correction ===")
print(f"LiDAR normal mean:   {n_lidar_mean}")
print(f"Camera normal mean:  {n_cam_mean}")
print(f"R_nominal predicted: {predicted}")
print(f"Correction angle Q:  {correction_deg:.1f}°")
print(f"\nCorrected R (LiDAR -> Camera):\n{np.round(R, 4)}")
print(f"det = {np.linalg.det(R):.4f}")

# Verify normal alignment
rotated_n  = (R @ normals_lidar.T).T
dots       = np.clip(np.einsum('ij,ij->i', rotated_n, normals_cam), -1.0, 1.0)
angles_deg = np.degrees(np.arccos(dots))
print(f"\nNormal alignment after correction:")
print(f"  Mean:   {angles_deg.mean():.2f}°")
print(f"  Median: {np.median(angles_deg):.2f}°")
print(f"  Max:    {angles_deg.max():.2f}°")

# ============================================================
# Step 3: Estimate scale S  (metres per square)
#
#   Board centre distance in LiDAR (metres) / board centre distance in camera (squares)
# ============================================================
dist_lidar  = np.linalg.norm(centers_lidar, axis=1)
dist_cam_sq = np.linalg.norm(centers_cam_sq, axis=1)
valid = (dist_cam_sq > 1.0) & (dist_lidar > 0.5) & (dist_lidar < 15.0)
S = float(np.median(dist_lidar[valid] / dist_cam_sq[valid]))

print(f"\n=== Scale ===")
print(f"Valid frames for S estimate: {valid.sum()}")
print(f"Square size S = {S * 100:.3f} cm")

# ============================================================
# Step 4: Estimate translation t
#
#   p_cam = R @ p_lidar + t
#   => t_i = c_cam_m_i - R @ c_lidar_i
#
#   c_cam_m = board centre in camera frame (metres)
# ============================================================
centers_cam_m = centers_cam_sq * S        # square units -> metres
rotated_lid   = (R @ centers_lidar.T).T   # R @ c_lidar  (metres)
t_estimates   = centers_cam_m - rotated_lid

t     = np.median(t_estimates, axis=0)
t_std = np.std(t_estimates,    axis=0)

print(f"\n=== Translation ===")
print(f"t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
print(f"std= [{t_std[0]:.4f}, {t_std[1]:.4f}, {t_std[2]:.4f}] m")
print(f"|t| = {np.linalg.norm(t):.3f} m  (camera-LiDAR distance)")

# Physical sanity check: where is the board in camera frame?
c_lidar_mean = np.mean(centers_lidar, axis=0)
c_cam_pred   = R @ c_lidar_mean + t
print(f"\nBoard centre in LiDAR:       {c_lidar_mean}")
print(f"Board centre in camera (m):  {c_cam_pred}")
c_cam_data   = np.median(centers_cam_m, axis=0)
print(f"Board centre in camera data: {c_cam_data}")

# ============================================================
# Step 5: Build RT 4x4 and save
# ============================================================
RT = np.eye(4)
RT[:3, :3] = R
RT[:3,  3] = t

print(f"\n=== Extrinsic RT (LiDAR -> Camera) ===")
print(np.round(RT, 4))

result = {
    "description": (
        "LiDAR->Camera extrinsic. R from minimal-rotation correction of R_nominal "
        "(cam_to_vr prior) using mean board normal from data. "
        "t from median board-centre correspondences."
    ),
    "estimated_square_size_m":  round(float(S), 6),
    "estimated_square_size_cm": round(float(S) * 100, 3),
    "R_lidar_to_cam":  R.tolist(),
    "t_lidar_to_cam":  t.tolist(),
    "RT_4x4":          RT.tolist(),
    "quality": {
        "num_frames":               len(data),
        "correction_angle_deg":     round(float(correction_deg), 2),
        "normal_error_mean_deg":    round(float(angles_deg.mean()),     3),
        "normal_error_median_deg":  round(float(np.median(angles_deg)), 3),
        "t_std_m":                  [round(float(x), 4) for x in t_std.tolist()],
        "camera_lidar_distance_m":  round(float(np.linalg.norm(t)), 4),
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {OUTPUT_JSON}")
print("Done!")
