"""
PZ9 - Labeled 3D scene: LiDAR + Board + Two Cameras
=====================================================
Produces a clear matplotlib figure with text labels showing:
  1. LiDAR sensor at origin
  2. Checkerboard board (position + normal)
  3. Left camera  (blue)
  4. Right camera (magenta)

Saves result to results/scene_labeled.png automatically.

Run:
  python visualize_scene.py
"""

import json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")          # change to "Agg" if no display available
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import laspy

os.makedirs("results", exist_ok=True)

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = "./9pz/experiment"
LIDAR_DIR      = os.path.join(DATA_DIR, "robosenseCapture")
CALIB_JSON     = "./results/lidar_camera_calib.json"
EXTRINSIC_JSON = "./results/extrinsic_lidar_to_cam.json"
LEFT_YML       = os.path.join(DATA_DIR, "leftImage.yml")
RIGHT_YML      = os.path.join(DATA_DIR, "rightImage.yml")

BOARD_SQUARES  = (7, 7)
SQUARE_SIZE_M  = 0.041

# ─── load calibration ─────────────────────────────────────────────────────────
with open(CALIB_JSON) as f:
    frames_data = json.load(f)
with open(EXTRINSIC_JSON) as f:
    ext = json.load(f)

R_l2c = np.array(ext["R_lidar_to_cam"])
t_l2c = np.array(ext["t_lidar_to_cam"])
S     = ext.get("estimated_square_size_m", SQUARE_SIZE_M)

R_c2l      = R_l2c.T
cam_L_orig = R_c2l @ (-t_l2c)
cam_L_axes = R_c2l

def read_K(yml):
    fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_READ)
    K  = fs.getNode("K").mat()
    sz = fs.getNode("sz")
    w, h = int(sz.at(0).real()), int(sz.at(1).real())
    fs.release()
    return K, (w, h)

K_L, sz_L = read_K(LEFT_YML)
K_R, sz_R = read_K(RIGHT_YML)

BASELINE_M = 0.12
cam_R_orig = cam_L_orig + cam_L_axes[:, 0] * BASELINE_M
cam_R_axes = cam_L_axes

# ─── LiDAR point cloud ───────────────────────────────────────────────────────
entry    = frames_data[0]
laz_path = os.path.join(LIDAR_DIR, entry["lidar_file"])
las      = laspy.read(laz_path)
pts_all  = np.vstack([las.x, las.y, las.z]).T
rng      = np.linalg.norm(pts_all, axis=1)
pts      = pts_all[(rng > 0.3) & (rng < 30.0)]

# ─── board ────────────────────────────────────────────────────────────────────
board_c = np.array(entry["board_center_lidar"])
board_n = np.array(entry["board_normal_lidar"])
if board_n[0] < 0:
    board_n = -board_n
board_n /= np.linalg.norm(board_n)

v_tmp = np.array([0., 0., 1.])
v_tmp -= np.dot(v_tmp, board_n) * board_n
if np.linalg.norm(v_tmp) < 1e-3:
    v_tmp = np.array([0., 1., 0.])
    v_tmp -= np.dot(v_tmp, board_n) * board_n
v1 = v_tmp / np.linalg.norm(v_tmp)
v2 = np.cross(board_n, v1)

n_sq_x = BOARD_SQUARES[0] + 1   # 8
n_sq_y = BOARD_SQUARES[1] + 1   # 8
hw = n_sq_x * S / 2.0
hh = n_sq_y * S / 2.0

# Board corners (rectangle outline)
board_corners = np.array([
    board_c + hw*v1 + hh*v2,
    board_c - hw*v1 + hh*v2,
    board_c - hw*v1 - hh*v2,
    board_c + hw*v1 - hh*v2,
])

# ─── helpers ──────────────────────────────────────────────────────────────────
def frustum_edges(orig, axes_c2l, K, img_sz, depth=1.2):
    w, h = img_sz
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    corners_cam = np.array([
        [(0-cx)/fx, (0-cy)/fy, 1.],
        [(w-cx)/fx, (0-cy)/fy, 1.],
        [(w-cx)/fx, (h-cy)/fy, 1.],
        [(0-cx)/fx, (h-cy)/fy, 1.],
    ]) * depth
    c = (axes_c2l @ corners_cam.T).T + orig
    # 4 edges from apex, 4 base edges
    segs = []
    for ci in c:
        segs.append((orig, ci))
    for i in range(4):
        segs.append((c[i], c[(i+1)%4]))
    return segs

def draw_frustum(ax, orig, axes, K, sz, color, depth=1.2):
    for a, b in frustum_edges(orig, axes, K, sz, depth):
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], color=color, linewidth=1.5)

def draw_axes(ax, orig, axes, length=0.4):
    colors = ['red', 'lime', 'blue']
    for i, c in enumerate(colors):
        end = orig + length * axes[:, i]
        ax.quiver(orig[0], orig[1], orig[2],
                  axes[0,i]*length, axes[1,i]*length, axes[2,i]*length,
                  color=c, linewidth=2, arrow_length_ratio=0.25)

def draw_checkerboard(ax, center, u_axis, v_axis, n_cols, n_rows, sq_size):
    polys_b, polys_w = [], []
    for i in range(n_cols):
        for j in range(n_rows):
            u0 = (i       - n_cols/2.) * sq_size
            u1 = (i + 1.0 - n_cols/2.) * sq_size
            w0 = (j       - n_rows/2.) * sq_size
            w1 = (j + 1.0 - n_rows/2.) * sq_size
            sq = np.array([
                center + u0*u_axis + w0*v_axis,
                center + u1*u_axis + w0*v_axis,
                center + u1*u_axis + w1*v_axis,
                center + u0*u_axis + w1*v_axis,
            ])
            if (i + j) % 2 == 0:
                polys_b.append(sq)
            else:
                polys_w.append(sq)
    if polys_b:
        ax.add_collection3d(Poly3DCollection(polys_b, facecolor='#111111', edgecolor='none', alpha=0.95))
    if polys_w:
        ax.add_collection3d(Poly3DCollection(polys_w, facecolor='#eeeeee', edgecolor='none', alpha=0.95))

# ═══════════════════════════════════════════════════════════════════════════════
# Figure: two panels — 3D perspective + top-down (XY)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')

# ── Panel 1: 3D perspective ───────────────────────────────────────────────────
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('#1a1a2e')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

# Downsampled point cloud
rng_idx = np.random.default_rng(0).choice(len(pts), min(4000, len(pts)), replace=False)
pt_sub  = pts[rng_idx]
sc = ax.scatter(pt_sub[:,0], pt_sub[:,1], pt_sub[:,2],
                c=pt_sub[:,2], cmap='viridis', s=1.0, alpha=0.35, zorder=1)

# [1] LiDAR sensor ─────────────────────────────────────────────────────────────
ax.scatter([0],[0],[0], c='orange', s=250, marker='s', zorder=5, linewidths=0)
draw_axes(ax, np.zeros(3), np.eye(3), length=0.35)
ax.text(0.0, 0.0, 0.55,
        "① LiDAR",
        color='orange', fontsize=13, fontweight='bold',
        bbox=dict(facecolor='#1a1a2e', edgecolor='orange', pad=3, alpha=0.85))

# [2] Checkerboard board ───────────────────────────────────────────────────────
draw_checkerboard(ax, board_c, v1, v2, n_sq_x, n_sq_y, S)
# Outline
bc = np.vstack([board_corners, board_corners[0]])
ax.plot(bc[:,0], bc[:,1], bc[:,2], color='yellow', linewidth=2, zorder=4)
# Normal arrow
ax.quiver(board_c[0], board_c[1], board_c[2],
          board_n[0]*0.35, board_n[1]*0.35, board_n[2]*0.35,
          color='#00e676', linewidth=2, arrow_length_ratio=0.3)
lbl_offset = board_n * 0.5
ax.text(board_c[0]+lbl_offset[0], board_c[1]+lbl_offset[1], board_c[2]+lbl_offset[2],
        "② Board\n(checkerboard)",
        color='yellow', fontsize=13, fontweight='bold',
        bbox=dict(facecolor='#1a1a2e', edgecolor='yellow', pad=3, alpha=0.85))

# [3a] Left camera ─────────────────────────────────────────────────────────────
draw_frustum(ax, cam_L_orig, cam_L_axes, K_L, sz_L, color='#00aaff', depth=1.2)
draw_axes(ax, cam_L_orig, cam_L_axes, length=0.25)
ax.scatter(*cam_L_orig, c='#00aaff', s=200, marker='^', zorder=6, linewidths=0)
off_L = cam_L_axes[:,1] * 0.4 + cam_L_axes[:,2] * (-0.3)
ax.text(cam_L_orig[0]+off_L[0], cam_L_orig[1]+off_L[1], cam_L_orig[2]+off_L[2],
        "③ Camera LEFT",
        color='#00aaff', fontsize=13, fontweight='bold',
        bbox=dict(facecolor='#1a1a2e', edgecolor='#00aaff', pad=3, alpha=0.85))

# [3b] Right camera ────────────────────────────────────────────────────────────
draw_frustum(ax, cam_R_orig, cam_R_axes, K_R, sz_R, color='#ff44ff', depth=1.0)
draw_axes(ax, cam_R_orig, cam_R_axes, length=0.25)
ax.scatter(*cam_R_orig, c='#ff44ff', s=200, marker='^', zorder=6, linewidths=0)
off_R = cam_R_axes[:,1] * 0.4 + cam_R_axes[:,2] * (-0.3)
ax.text(cam_R_orig[0]+off_R[0], cam_R_orig[1]+off_R[1], cam_R_orig[2]+off_R[2],
        "③ Camera RIGHT",
        color='#ff44ff', fontsize=13, fontweight='bold',
        bbox=dict(facecolor='#1a1a2e', edgecolor='#ff44ff', pad=3, alpha=0.85))

# Stereo baseline
ax.plot([cam_L_orig[0], cam_R_orig[0]],
        [cam_L_orig[1], cam_R_orig[1]],
        [cam_L_orig[2], cam_R_orig[2]],
        color='#cc44cc', linewidth=1.5, linestyle='--')

ax.set_xlabel("X (m)", color='white', labelpad=6)
ax.set_ylabel("Y (m)", color='white', labelpad=6)
ax.set_zlabel("Z (m)", color='white', labelpad=6)
ax.set_title("3D scene (perspective)", color='white', fontsize=13, pad=10)
ax.view_init(elev=25, azim=-60)

# ── Panel 2: top-down XY view ─────────────────────────────────────────────────
ax2 = fig.add_subplot(122)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')
ax2.spines[:].set_edgecolor('white')

# Point cloud top-down
ax2.scatter(pt_sub[:,0], pt_sub[:,1], c=pt_sub[:,2], cmap='viridis',
            s=0.8, alpha=0.3, zorder=1)

# LiDAR
ax2.scatter([0],[0], c='orange', s=300, marker='s', zorder=5, linewidths=0)
ax2.annotate("① LiDAR", xy=(0,0), xytext=(0.3, 0.3),
             fontsize=12, fontweight='bold', color='orange',
             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
             bbox=dict(facecolor='#1a1a2e', edgecolor='orange', pad=3))

# Board (top-down projection = outline in XY)
ax2.plot(np.append(board_corners[:,0], board_corners[0,0]),
         np.append(board_corners[:,1], board_corners[0,1]),
         color='yellow', linewidth=2.5, zorder=4)
ax2.scatter([board_c[0]], [board_c[1]], c='yellow', s=150, marker='*', zorder=5)
ax2.annotate("② Board", xy=(board_c[0], board_c[1]),
             xytext=(board_c[0]+0.4, board_c[1]+0.4),
             fontsize=12, fontweight='bold', color='yellow',
             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
             bbox=dict(facecolor='#1a1a2e', edgecolor='yellow', pad=3))

# Left camera
ax2.scatter([cam_L_orig[0]], [cam_L_orig[1]], c='#00aaff', s=250, marker='^', zorder=6)
ax2.annotate("③ Cam LEFT", xy=(cam_L_orig[0], cam_L_orig[1]),
             xytext=(cam_L_orig[0]-0.5, cam_L_orig[1]+0.5),
             fontsize=12, fontweight='bold', color='#00aaff',
             arrowprops=dict(arrowstyle='->', color='#00aaff', lw=1.5),
             bbox=dict(facecolor='#1a1a2e', edgecolor='#00aaff', pad=3))

# Right camera
ax2.scatter([cam_R_orig[0]], [cam_R_orig[1]], c='#ff44ff', s=250, marker='^', zorder=6)
ax2.annotate("③ Cam RIGHT", xy=(cam_R_orig[0], cam_R_orig[1]),
             xytext=(cam_R_orig[0]+0.4, cam_R_orig[1]-0.5),
             fontsize=12, fontweight='bold', color='#ff44ff',
             arrowprops=dict(arrowstyle='->', color='#ff44ff', lw=1.5),
             bbox=dict(facecolor='#1a1a2e', edgecolor='#ff44ff', pad=3))

# Stereo baseline
ax2.plot([cam_L_orig[0], cam_R_orig[0]],
         [cam_L_orig[1], cam_R_orig[1]],
         color='#cc44cc', linewidth=2, linestyle='--', label=f'baseline ~{BASELINE_M*100:.0f} cm')

ax2.set_xlabel("X (m)", color='white')
ax2.set_ylabel("Y (m)", color='white')
ax2.set_title("Top-down view (XY)", color='white', fontsize=13)
ax2.set_aspect('equal')
ax2.grid(True, color='#333355', linewidth=0.5)
ax2.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='white', fontsize=10)

# ─── master title + legend strip ─────────────────────────────────────────────
fig.suptitle(
    "PZ9 — LiDAR ↔ Stereo Camera Calibration Scene",
    color='white', fontsize=16, fontweight='bold', y=0.97
)

# Legend strip at bottom
legend_txt = (
    "  ■ orange square = ① LiDAR sensor   "
    "  ■ yellow rect   = ② Checkerboard board   "
    "  ▲ blue triangle = ③ Camera LEFT   "
    "  ▲ magenta ▲     = ③ Camera RIGHT  "
)
fig.text(0.5, 0.01, legend_txt, ha='center', color='#cccccc', fontsize=10,
         bbox=dict(facecolor='#0d0d1a', edgecolor='#444466', pad=4))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_path = "results/scene_labeled.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
print(f"  ① LiDAR       : [0, 0, 0]")
print(f"  ② Board center: {np.round(board_c, 3)}")
print(f"  ③ Cam LEFT    : {np.round(cam_L_orig, 3)}")
print(f"  ③ Cam RIGHT   : {np.round(cam_R_orig, 3)}")

plt.show()
