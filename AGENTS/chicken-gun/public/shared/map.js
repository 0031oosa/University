// Farm arena. Shared between server (hit detection, egg physics) and client (rendering, collision).
// Every solid is an axis-aligned box: { min:[x,y,z], max:[x,y,z], type }.

export const ARENA = 36; // half-size of the playable square
export const BOXES = [];

function box(cx, cz, w, d, h, y = 0, type = 'crate') {
  BOXES.push({ min: [cx - w / 2, y, cz - d / 2], max: [cx + w / 2, y + h, cz + d / 2], type });
}
const crate = (x, z, y = 0) => box(x, z, 1.2, 1.2, 1.2, y, 'crate');
const hay = (x, z, rotated = false, y = 0) =>
  rotated ? box(x, z, 1.1, 2.2, 1.1, y, 'hay') : box(x, z, 2.2, 1.1, 1.1, y, 'hay');

// --- Outer fence ---
const A = ARENA;
box(0, -A - 0.5, 2 * A + 2, 1, 4, 0, 'wall');
box(0, A + 0.5, 2 * A + 2, 1, 4, 0, 'wall');
box(-A - 0.5, 0, 1, 2 * A, 4, 0, 'wall');
box(A + 0.5, 0, 1, 2 * A, 4, 0, 'wall');

// --- Barn (center) with four doors, a hayloft and stairs ---
export const BARN = { hw: 8, hd: 6, h: 5 };
for (const s of [-1, 1]) {
  box(-4.875, s * 6, 6.75, 0.5, 5, 0, 'barn');
  box(4.875, s * 6, 6.75, 0.5, 5, 0, 'barn');
  box(0, s * 6, 3, 0.5, 2, 3, 'barn');
  box(s * 8, -3.75, 0.5, 4.5, 5, 0, 'barn');
  box(s * 8, 3.75, 0.5, 4.5, 5, 0, 'barn');
  box(s * 8, 0, 0.5, 3, 2, 3, 'barn');
}
box(0, 0, 17, 13, 0.4, 5, 'roof');
box(4.875, 0, 5.75, 11.5, 0.3, 2.4, 'plank'); // hayloft, top at 2.7
for (let i = 1; i <= 6; i++) box(-1.6 + (i - 0.5) * 0.6, -4.75, 0.6, 2, 0.45 * i, 0, 'plank');
hay(5, -4.5, false, 2.7);
hay(6.5, 4.5, true, 2.7);
crate(-6.5, 4.5); crate(-6.5, 3.3); crate(-6.5, 4.5, 1.2);
hay(-5.5, -1.5, true);

// --- Corner watchtowers ---
for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
  const cx = sx * 27, cz = sz * 27, PH = 3.5;
  box(cx, cz, 5, 5, 0.3, PH, 'plank'); // platform, top at 3.8
  for (const px of [-1, 1]) for (const pz of [-1, 1]) box(cx + px * 2.3, cz + pz * 2.3, 0.4, 0.4, PH, 0, 'post');
  box(cx + sx * 2.4, cz, 0.2, 5, 0.8, PH + 0.3, 'plank');
  box(cx, cz + sz * 2.4, 5, 0.2, 0.8, PH + 0.3, 'plank');
  const edge = cx - sx * 2.5;
  for (let k = 1; k <= 8; k++) box(edge - sx * (k - 0.5) * 0.6, cz, 0.6, 1.6, 3.8 - (k - 1) * 0.475, 0, 'plank');
}

// --- Silos ---
export const SILOS = [[-16, 22], [16, -22]];
for (const [x, z] of SILOS) box(x, z, 4, 4, 9, 0, 'silo');

// --- Cover ---
crate(-14, -12); crate(-14, -12, 1.2); crate(-12.8, -12);
crate(14, 12); crate(14, 12, 1.2); crate(12.8, 12);
crate(-20, 14); crate(20, -14); crate(-6, -18); crate(6, 18);
crate(-25, -8); crate(25, 8); crate(-10, 28); crate(10, -28);
hay(-12, 0, true); hay(12, 0, true);
hay(0, -11); hay(0, 11);
hay(-22, -4, true); hay(22, 4, true);
hay(-24, 8); hay(24, -8);
hay(-8, 24); hay(8, -24); hay(-8, 24, false, 1.1); hay(8, -24, false, 1.1);
box(-18, -20, 6, 0.8, 1.3, 0, 'stone');
box(18, 20, 6, 0.8, 1.3, 0, 'stone');
box(-29, 6, 0.8, 6, 1.3, 0, 'stone');
box(29, -6, 0.8, 6, 1.3, 0, 'stone');
box(6, -30, 6, 0.8, 1.3, 0, 'stone');
box(-6, 30, 6, 0.8, 1.3, 0, 'stone');

// --- Spawns & pickups ---
export const SPAWNS = [
  [-30, 0, 0], [30, 0, 0], [0, 0, -30], [0, 0, 30],
  [-15, 0, -30], [15, 0, 30], [-31, 0, -16], [31, 0, 16],
  [-10, 0, -25], [10, 0, 25], [0, 0, 0], [-25, 0, 18], [25, 0, -18],
  [-3, 0, 2.5], [-18, 0, 5], [18, 0, -5],
];

export const PICKUPS = [
  [0, 0, -16], [0, 0, 16], [-19, 0, 0], [19, 0, 0],
  [5, 2.7, 0], [27, 3.8, 27], [-27, 3.8, -27],
];

// --- Ray helpers (d must be normalized) ---
export function rayBox(o, d, b) {
  let tmin = 0, tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-9) {
      if (o[i] < b.min[i] || o[i] > b.max[i]) return Infinity;
    } else {
      let t1 = (b.min[i] - o[i]) / d[i], t2 = (b.max[i] - o[i]) / d[i];
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return Infinity;
    }
  }
  return tmin;
}

export function raycastWorld(o, d, maxT = 1000) {
  let t = maxT;
  if (d[1] < 0) { const tg = -o[1] / d[1]; if (tg >= 0 && tg < t) t = tg; }
  for (const b of BOXES) { const tb = rayBox(o, d, b); if (tb < t) t = tb; }
  return t;
}

export function raySphere(o, d, c, r) {
  const ox = o[0] - c[0], oy = o[1] - c[1], oz = o[2] - c[2];
  const b = ox * d[0] + oy * d[1] + oz * d[2];
  const cc = ox * ox + oy * oy + oz * oz - r * r;
  const disc = b * b - cc;
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  if (t >= 0) return t;
  return cc < 0 ? 0 : Infinity;
}

// Chicken hitboxes: body sphere + head sphere (head is in front, along facing -Z rotated by ry).
export function chickenHit(o, d, pos, ry) {
  const head = [pos[0] - Math.sin(ry) * 0.28, pos[1] + 1.05, pos[2] - Math.cos(ry) * 0.28];
  const th = raySphere(o, d, head, 0.32);
  const tb = raySphere(o, d, [pos[0], pos[1] + 0.58, pos[2]], 0.55);
  if (th <= tb) return { t: th, head: th < Infinity };
  return { t: tb, head: false };
}
