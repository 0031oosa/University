"""
PZ9 - 3D visualization: LiDAR + Board + Two Cameras
====================================================
Interactive Polyscope 3D window.
Text labels float directly next to each object in the scene
and move with it as you rotate the view:
  ① LiDAR sensor  (orange box at origin)
  ② Board          (black/white checkerboard)
  ③ Camera LEFT   (blue frustum)
  ③ Camera RIGHT  (magenta frustum)

A separate OpenCV window shows the selected calibration frame from
both cameras with checkerboard detection overlay.

Run:
  python visualize_3d.py
"""

import json, os, threading
import numpy as np
import laspy
import cv2
import polyscope as ps
import polyscope.imgui as psim

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = "./9pz/experiment"
LIDAR_DIR      = os.path.join(DATA_DIR, "robosenseCapture")
CALIB_JSON     = "./results/lidar_camera_calib.json"
EXTRINSIC_JSON = "./results/extrinsic_lidar_to_cam.json"
LEFT_YML       = os.path.join(DATA_DIR, "leftImage.yml")
RIGHT_YML      = os.path.join(DATA_DIR, "rightImage.yml")

BOARD_SQUARES = (7, 7)
SQUARE_SIZE_M = 0.041

LEFT_VIDEO  = os.path.join(DATA_DIR, "xt1.021.003.left.avi")
RIGHT_VIDEO = os.path.join(DATA_DIR, "xt1.021.003.right.avi")
VID_PANEL_H = 360   # pixel height of each camera panel in the detection window

CAMERA_TARGET_FRAME = 234  # video frame to show in the camera window

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

def read_K(yml):
    fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_READ)
    K  = fs.getNode("K").mat()
    sz = fs.getNode("sz")
    w, h = int(sz.at(0).real()), int(sz.at(1).real())
    fs.release()
    return K, (w, h)

def read_D(yml):
    fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_READ)
    try:
        D = fs.getNode("D").mat()
    except Exception:
        D = np.zeros((5, 1))
    fs.release()
    return D

K_L, sz_L = read_K(LEFT_YML)
K_R, sz_R = read_K(RIGHT_YML)
D_L = read_D(LEFT_YML)
D_R = read_D(RIGHT_YML)

# ─── LiDAR point cloud ────────────────────────────────────────────────────────
entry    = frames_data[0]
laz_path = os.path.join(LIDAR_DIR, entry["lidar_file"])
las      = laspy.read(laz_path)
pts_all  = np.vstack([las.x, las.y, las.z]).T
rng      = np.linalg.norm(pts_all, axis=1)
pts      = pts_all[(rng > 0.3) & (rng < 30.0)]

# ─── board geometry ───────────────────────────────────────────────────────────
board_c = np.array([-1.340, 5.010, -0.501])  # actual board pos from LiDAR cloud
board_n = cam_L_orig - board_c   # normal faces camera
board_n /= np.linalg.norm(board_n)

# look-at from cam toward actual board
_f = board_c - cam_L_orig; _f /= np.linalg.norm(_f)
_r = np.cross(_f, np.array([0.,0.,1.])); _r /= np.linalg.norm(_r)
_u = np.cross(_f, _r)
cam_L_axes = np.column_stack([_r, _u, _f])

BASELINE_M = 0.12
cam_R_orig = cam_L_orig + cam_L_axes[:, 0] * BASELINE_M
cam_R_axes = cam_L_axes

v_tmp = np.array([0., 0., 1.])
v_tmp -= np.dot(v_tmp, board_n) * board_n
if np.linalg.norm(v_tmp) < 1e-3:
    v_tmp = np.array([0., 1., 0.])
    v_tmp -= np.dot(v_tmp, board_n) * board_n
v1 = v_tmp / np.linalg.norm(v_tmp)
v2 = np.cross(board_n, v1)

n_sq_x = BOARD_SQUARES[0] + 1
n_sq_y = BOARD_SQUARES[1] + 1
hw = n_sq_x * S / 2.0
hh = n_sq_y * S / 2.0

# ─── camera detection window ──────────────────────────────────────────────────
def detect_and_draw(img, K, D, board_pattern, sq_size, label, frame_idx, ts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board_pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    out = img.copy()
    mse_str = "?.??px"
    if found:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        nr, nc = board_pattern
        objp = np.zeros((nr * nc, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nr, 0:nc].T.reshape(-1, 2) * sq_size
        ret, rvec_s, tvec_s = cv2.solvePnP(objp, corners, K, D)
        if ret:
            imgp, _ = cv2.projectPoints(objp, rvec_s, tvec_s, K, D)
            mse = float(np.sqrt(np.mean(
                (corners.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2)))
            mse_str = f"{mse:.2f}px"
            cv2.drawFrameAxes(out, K, D, rvec_s, tvec_s, sq_size * 3, 3)
        pts2d = corners.reshape(-1, 2)
        x1, y1 = int(pts2d[:, 0].min()), int(pts2d[:, 1].min())
        x2, y2 = int(pts2d[:, 0].max()), int(pts2d[:, 1].max())
        pad = 12
        cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                      (255, 255, 0), 2)  # cyan in BGR
        for c in pts2d:
            cv2.circle(out, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
        c0 = pts2d[0]
        cv2.drawMarker(out, (int(c0[0]), int(c0[1])),
                       (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(out, f"{label} ts={ts} frame={frame_idx} mse={mse_str}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def build_camera_panel():
    cam_entry = min(frames_data,
                    key=lambda e: abs(e["left_frame_idx"] - CAMERA_TARGET_FRAME))
    cap_l = cv2.VideoCapture(LEFT_VIDEO)
    cap_r = cv2.VideoCapture(RIGHT_VIDEO)
    ts   = cam_entry["time_ms"]
    fi_l = cam_entry["left_frame_idx"]
    fi_r = cam_entry["right_frame_idx"]
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, fi_l)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, fi_r)
    ret_l, img_l = cap_l.read()
    ret_r, img_r = cap_r.read()
    cap_l.release()
    cap_r.release()
    if not (ret_l and ret_r):
        print("[warn] Could not read camera frames — skipping camera window")
        return None
    pat   = BOARD_SQUARES
    out_l = detect_and_draw(img_l, K_L, D_L, pat, S, "LEFT",  fi_l, ts)
    out_r = detect_and_draw(img_r, K_R, D_R, pat, S, "RIGHT", fi_r, ts - 1)

    def resize_h(img, h):
        r = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), h))

    return np.hstack([resize_h(out_l, VID_PANEL_H), resize_h(out_r, VID_PANEL_H)])


_cam_stop = threading.Event()


def _camera_win_fn(panel):
    cv2.namedWindow("Camera Detection", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Camera Detection", panel)
    while not _cam_stop.is_set():
        key = cv2.waitKey(100)
        if key in (27, ord('q'), ord('Q')):
            _cam_stop.set()
    cv2.destroyAllWindows()


panel = build_camera_panel()
if panel is not None:
    threading.Thread(target=_camera_win_fn, args=(panel,), daemon=True).start()

# ─── helpers ──────────────────────────────────────────────────────────────────
def make_checkerboard(center, u_axis, v_axis, n_cols, n_rows, sq_size):
    verts, faces, colors = [], [], []
    vi = 0
    for i in range(n_cols):
        for j in range(n_rows):
            u0 = (i       - n_cols/2.) * sq_size
            u1 = (i + 1.0 - n_cols/2.) * sq_size
            w0 = (j       - n_rows/2.) * sq_size
            w1 = (j + 1.0 - n_rows/2.) * sq_size
            p = [center + u0*u_axis + w0*v_axis,
                 center + u1*u_axis + w0*v_axis,
                 center + u1*u_axis + w1*v_axis,
                 center + u0*u_axis + w1*v_axis]
            verts.extend(p)
            faces.extend([[vi,vi+1,vi+2],[vi,vi+2,vi+3]])
            col = (0.08,0.08,0.08) if (i+j)%2==0 else (0.95,0.95,0.95)
            colors.extend([col, col])
            vi += 4
    return np.array(verts,float), np.array(faces,int), np.array(colors,float)

def make_frustum(orig, axes_c2l, K, img_sz, depth=1.0):
    w, h   = img_sz
    fx,fy  = K[0,0], K[1,1]
    cx,cy  = K[0,2], K[1,2]
    cc = np.array([
        [(0-cx)/fx,(0-cy)/fy,1.],
        [(w-cx)/fx,(0-cy)/fy,1.],
        [(w-cx)/fx,(h-cy)/fy,1.],
        [(0-cx)/fx,(h-cy)/fy,1.],
    ]) * depth
    c = (axes_c2l @ cc.T).T + orig
    nodes = np.vstack([orig, c])
    edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]])
    return nodes, edges

def make_axes(orig, axes, length=0.5):
    nodes = np.array([
        orig, orig+length*axes[:,0],
        orig, orig+length*axes[:,1],
        orig, orig+length*axes[:,2],
    ], float)
    edges  = np.array([[0,1],[2,3],[4,5]])
    colors = np.array([[1,0,0],[1,0,0],[0,.8,0],[0,.8,0],[0,0,1],[0,0,1]], float)
    return nodes, edges, colors

def make_box(center, size=0.05):
    h = size/2.; c = center
    v = np.array([c+[-h,-h,-h],c+[h,-h,-h],c+[h,h,-h],c+[-h,h,-h],
                  c+[-h,-h, h],c+[h,-h, h],c+[h,h, h],c+[-h,h, h]], float)
    e = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                  [0,4],[1,5],[2,6],[3,7]])
    return v, e

def make_grid(size=8, step=1.0, z=0.0):
    coords = np.linspace(-size, size, int(2*size/step)+1)
    nodes, edges, idx = [], [], 0
    for c in coords:
        nodes+=[[c,-size,z],[c,size,z]]; edges.append([idx,idx+1]); idx+=2
        nodes+=[[-size,c,z],[size,c,z]]; edges.append([idx,idx+1]); idx+=2
    return np.array(nodes,float), np.array(edges)

# ═══════════════════════════════════════════════════════════════════════════════
# Polyscope scene
# ═══════════════════════════════════════════════════════════════════════════════
ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("none")

# Ground grid
gn, ge = make_grid(size=8, z=pts[:,2].min())
ps.register_curve_network("Ground grid", gn, ge, radius=0.002).set_color((0.25,0.65,0.25))

# LiDAR cloud
pc = ps.register_point_cloud("LiDAR cloud", pts, radius=0.001)
h_n = (pts[:,2]-pts[:,2].min()) / (pts[:,2].max()-pts[:,2].min()+1e-6)
pc.add_scalar_quantity("height", h_n, enabled=True, cmap="viridis")

# ① LiDAR sensor — orange wireframe box at origin
bv, be = make_box(np.zeros(3), size=0.12)
lb = ps.register_curve_network("(1) LiDAR sensor", bv, be, radius=0.003)
lb.set_color((1.0, 0.5, 0.0))

an, ae, ac = make_axes(np.zeros(3), np.eye(3))
la = ps.register_curve_network("LiDAR frame", an, ae, radius=0.004)
la.add_color_quantity("c", ac, defined_on="nodes", enabled=True)

# ② Board — checkerboard mesh + yellow outline + normal arrow
cb_v, cb_f, cb_c = make_checkerboard(board_c, v1, v2, n_sq_x, n_sq_y, S)
bm = ps.register_surface_mesh("(2) Board", cb_v, cb_f)
bm.add_color_quantity("sq", cb_c, defined_on="faces", enabled=True)

board_corners = np.array([board_c+hw*v1+hh*v2, board_c-hw*v1+hh*v2,
                           board_c-hw*v1-hh*v2, board_c+hw*v1-hh*v2,
                           board_c+hw*v1+hh*v2])
outline = ps.register_curve_network("Board outline", board_corners,
                                    np.array([[i,i+1] for i in range(4)]), radius=0.002)
outline.set_color((1.0, 1.0, 0.0))

bn_nodes = np.array([board_c, board_c + 0.35*board_n], float)
bn = ps.register_curve_network("Board normal", bn_nodes, np.array([[0,1]]), radius=0.002)
bn.set_color((0.0, 0.85, 0.35))

# Board inlier LiDAR points
bd = float(np.dot(board_n, board_c))
dp = np.abs(pts @ board_n - bd)
pr = pts - board_c
in_r = (np.abs(pr @ v1) < hw*1.1) & (np.abs(pr @ v2) < hh*1.1)
pts_b = pts[(dp < 0.02) & in_r]
if len(pts_b):
    ps.register_point_cloud("Board LiDAR pts", pts_b, radius=0.004).set_color((1.0,0.2,0.0))

# ③ Camera LEFT — blue
an, ae, ac = make_axes(cam_L_orig, cam_L_axes)
ca = ps.register_curve_network("(3) Camera LEFT frame", an, ae, radius=0.004)
ca.add_color_quantity("c", ac, defined_on="nodes", enabled=True)

fn, fe = make_frustum(cam_L_orig, cam_L_axes, K_L, sz_L, depth=1.5)
fl = ps.register_curve_network("(3) Camera LEFT frustum", fn, fe, radius=0.002)
fl.set_color((0.0, 0.55, 1.0))

bv, be = make_box(cam_L_orig, size=0.08)
ps.register_curve_network("Camera LEFT body", bv, be, radius=0.003).set_color((0.0,0.55,1.0))

# ③ Camera RIGHT — magenta
an, ae, ac = make_axes(cam_R_orig, cam_R_axes)
cr = ps.register_curve_network("(3) Camera RIGHT frame", an, ae, radius=0.004)
cr.add_color_quantity("c", ac, defined_on="nodes", enabled=True)

fn, fe = make_frustum(cam_R_orig, cam_R_axes, K_R, sz_R, depth=1.0)
fr = ps.register_curve_network("(3) Camera RIGHT frustum", fn, fe, radius=0.002)
fr.set_color((1.0, 0.27, 1.0))

bv, be = make_box(cam_R_orig, size=0.08)
ps.register_curve_network("Camera RIGHT body", bv, be, radius=0.003).set_color((1.0,0.27,1.0))

# Stereo baseline
bl = ps.register_curve_network("Stereo baseline",
                                np.array([cam_L_orig, cam_R_orig]),
                                np.array([[0,1]]), radius=0.002)
bl.set_color((0.7, 0.0, 0.9))

# ─── Floating 3D labels ────────────────────────────────────────────────────────
_FL = 1|2|4|8|32|64|128|256

_LABELS = [
    (np.array([0., 0., 0.25]),          "① LiDAR",        (1.00, 0.55, 0.00, 1.0)),
    (board_c + board_n * 0.35,          "② Board",         (1.00, 1.00, 0.00, 1.0)),
    (cam_L_orig + cam_L_axes[:,2]*0.25, "③ Camera LEFT",   (0.00, 0.67, 1.00, 1.0)),
    (cam_R_orig + cam_R_axes[:,2]*0.25, "③ Camera RIGHT",  (1.00, 0.27, 1.00, 1.0)),
]

def _world_to_screen(world_pos):
    """Project a 3D world point to 2D window pixel coordinates."""
    try:
        cam    = ps.get_view_camera_parameters()   # correct polyscope API
        view   = cam.get_view_mat()                # 4×4 view matrix, ready to use
        fov_v  = np.radians(cam.get_fov_vertical_deg())
        aspect = cam.get_aspect()
        w, h   = ps.get_window_size()

        near, far = 0.01, 1000.0
        t = np.tan(fov_v / 2.0)
        proj = np.array([
            [1/(aspect*t), 0,   0,                          0                      ],
            [0,            1/t, 0,                          0                      ],
            [0,            0,  -(far+near)/(far-near), -2*far*near/(far-near)],
            [0,            0,  -1,                          0                      ],
        ])

        clip = proj @ view @ np.array([*world_pos, 1.0])
        if clip[3] <= 0:
            return None
        ndc = clip[:3] / clip[3]
        sx = ( ndc[0] + 1) * 0.5 * w
        sy = (-ndc[1] + 1) * 0.5 * h
        if not (0 < sx < w and 0 < sy < h):
            return None
        return float(sx), float(sy)
    except Exception as e:
        return None

_MIN_SEP = 90   # minimum pixel distance between labels

def _label_callback():
    # 1. Project all positions
    items = []
    for pos, text, col in _LABELS:
        sc = _world_to_screen(pos)
        if sc is not None:
            items.append([list(sc), text, col])

    # 2. Deconflict: push overlapping labels apart (10 relaxation steps)
    for _ in range(10):
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ax, ay = items[a][0]
                bx, by = items[b][0]
                dx, dy = bx - ax, by - ay
                dist = (dx*dx + dy*dy) ** 0.5
                if 0 < dist < _MIN_SEP:
                    push = (_MIN_SEP - dist) / 2.0
                    nx, ny = dx / dist, dy / dist
                    items[a][0][0] -= nx * push
                    items[a][0][1] -= ny * push
                    items[b][0][0] += nx * push
                    items[b][0][1] += ny * push

    # 3. Draw
    for i, (sc, text, col) in enumerate(items):
        psim.SetNextWindowPos(tuple(sc))
        psim.SetNextWindowBgAlpha(0.0)
        psim.Begin(f"##lbl{i}", True, _FL)
        psim.TextColored(col, text)
        psim.End()

ps.set_user_callback(_label_callback)

print("\nScene elements:")
print(f"  ① LiDAR       : [0, 0, 0]   (orange box)")
print(f"  ② Board center: {np.round(board_c, 3)}  (yellow outline)")
print(f"  ③ Cam LEFT    : {np.round(cam_L_orig, 3)}  (blue)")
print(f"  ③ Cam RIGHT   : {np.round(cam_R_orig, 3)}  (magenta)")
print("\nControls: left-drag=rotate  scroll=zoom  right-drag=pan\n")

ps.show()
_cam_stop.set()
