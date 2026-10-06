import * as THREE from 'three';
import { BOXES, PICKUPS, SILOS, ARENA, raycastWorld, chickenHit } from './shared/map.js';
import { WEAPONS, WEAPON_LABELS, EGG, PLAYER, SCORE_LIMIT, RESPAWN_MS } from './shared/weapons.js';

// =====================================================================
// Settings & DOM helpers
// =====================================================================
const $ = (id) => document.getElementById(id);
const COLORS = ['#ffffff', '#ffd23f', '#8b5a2b', '#333333', '#e63946', '#4ea8de', '#57cc99', '#ff8fab', '#9b5de5', '#ff9f1c'];
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};
const settings = {
  name: store.get('cg_name', ''),
  color: store.get('cg_color', COLORS[Math.floor(Math.random() * COLORS.length)]),
  sens: Number(store.get('cg_sens', 1)) || 1,
  vol: Number(store.get('cg_vol', 0.6)),
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const setText = (el, txt) => { if (el._t !== txt) { el._t = txt; el.textContent = txt; } };
const setHTML = (el, html) => { if (el._h !== html) { el._h = html; el.innerHTML = html; } };
const nowSec = () => performance.now() / 1000;
const rnd = (a, b) => a + Math.random() * (b - a);

// =====================================================================
// Renderer, scenes, cameras, lights
// =====================================================================
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.autoClear = false;
renderer.domElement.id = 'game';
document.body.prepend(renderer.domElement);
const canvas = renderer.domElement;

const SKY = 0x8fcbef;
const scene = new THREE.Scene();
scene.background = new THREE.Color(SKY);
scene.fog = new THREE.Fog(SKY, 50, 160);

const BASE_FOV = 75;
const camera = new THREE.PerspectiveCamera(BASE_FOV, innerWidth / innerHeight, 0.05, 500);
camera.rotation.order = 'YXZ';

const vmScene = new THREE.Scene();
const vmCamera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 10);

scene.add(new THREE.HemisphereLight(0xd8ecff, 0x5a7b3a, 1.7));
const sun = new THREE.DirectionalLight(0xfff1d6, 2.9);
sun.position.set(25, 45, 15);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 1, far: 120 });
sun.shadow.bias = -0.0006;
sun.shadow.normalBias = 0.03;
scene.add(sun);

vmScene.add(new THREE.HemisphereLight(0xffffff, 0x666655, 2));
const vmSun = new THREE.DirectionalLight(0xffffff, 1.5);
vmSun.position.set(1, 2, 1);
vmScene.add(vmSun);

const flashLight = new THREE.PointLight(0xffc870, 0, 12, 2);
scene.add(flashLight);
const flashState = { peak: 0, dur: 1, t: 1 };
function lightFlash(pos, intensity, dur, distance) {
  flashLight.position.copy(pos);
  flashLight.distance = distance;
  flashState.peak = intensity; flashState.dur = dur; flashState.t = 0;
}

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = vmCamera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  vmCamera.updateProjectionMatrix();
});

// =====================================================================
// Procedural textures & materials
// =====================================================================
function canvasTex(size, draw) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  draw(c.getContext('2d'), size);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
function speckle(g, s, colors, n, w = 2, h = 2) {
  for (let i = 0; i < n; i++) {
    g.fillStyle = colors[(Math.random() * colors.length) | 0];
    g.fillRect(Math.random() * s, Math.random() * s, w, h);
  }
}
function lines(g, s, color, step, width, vertical) {
  g.fillStyle = color;
  for (let p = 0; p < s; p += step) vertical ? g.fillRect(p, 0, width, s) : g.fillRect(0, p, s, width);
}

const TEX = {
  grass: canvasTex(256, (g, s) => {
    g.fillStyle = '#5b9a3c'; g.fillRect(0, 0, s, s);
    for (let i = 0; i < 4000; i++) {
      g.fillStyle = ['#4f8a33', '#67a845', '#7cb85a', '#53913a'][(Math.random() * 4) | 0];
      g.fillRect(Math.random() * s, Math.random() * s, 2, 3 + Math.random() * 5);
    }
  }),
  crate: canvasTex(128, (g, s) => {
    g.fillStyle = '#b07a3e'; g.fillRect(0, 0, s, s);
    speckle(g, s, ['#a06c34', '#bd8748'], 600, 6, 1);
    lines(g, s, '#8a5a2a', 32, 2, false);
    g.strokeStyle = '#6e4520'; g.lineWidth = 14; g.strokeRect(7, 7, s - 14, s - 14);
    g.lineWidth = 12; g.beginPath(); g.moveTo(12, 12); g.lineTo(s - 12, s - 12); g.stroke();
  }),
  hay: canvasTex(128, (g, s) => {
    g.fillStyle = '#e2c25b'; g.fillRect(0, 0, s, s);
    g.lineWidth = 1.5;
    for (let i = 0; i < 500; i++) {
      g.strokeStyle = ['#c8a33e', '#f2dc85', '#d4b04a'][(Math.random() * 3) | 0];
      const x = Math.random() * s, y = Math.random() * s, a = rnd(-0.4, 0.4), l = rnd(6, 18);
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke();
    }
    g.fillStyle = '#7d6224'; g.fillRect(s * 0.28, 0, 4, s); g.fillRect(s * 0.7, 0, 4, s);
  }),
  barn: canvasTex(128, (g, s) => {
    g.fillStyle = '#a8322d'; g.fillRect(0, 0, s, s);
    speckle(g, s, ['#9a2b27', '#b53a34'], 500, 1, 6);
    lines(g, s, '#7d211d', 16, 2, true);
  }),
  plank: canvasTex(128, (g, s) => {
    g.fillStyle = '#9b7445'; g.fillRect(0, 0, s, s);
    speckle(g, s, ['#8c683c', '#a98150'], 600, 8, 1);
    lines(g, s, '#6e4f2c', 32, 2, false);
  }),
  stone: canvasTex(128, (g, s) => {
    g.fillStyle = '#6b6b65'; g.fillRect(0, 0, s, s);
    for (let row = 0; row < 8; row++) for (let col = -1; col < 4; col++) {
      const v = 130 + ((Math.random() * 30) | 0);
      g.fillStyle = `rgb(${v},${v},${v - 6})`;
      g.fillRect(col * 32 + (row % 2) * 16 + 2, row * 16 + 2, 28, 12);
    }
  }),
  wall: canvasTex(128, (g, s) => {
    g.fillStyle = '#7a5530'; g.fillRect(0, 0, s, s);
    speckle(g, s, ['#6d4a29', '#86603a'], 500, 1, 8);
    lines(g, s, '#4f3519', 21, 3, true);
    g.fillStyle = '#5a3d20'; g.fillRect(0, s * 0.2, s, 8); g.fillRect(0, s * 0.75, s, 8);
  }),
  roof: canvasTex(128, (g, s) => {
    g.fillStyle = '#5e2a24'; g.fillRect(0, 0, s, s);
    lines(g, s, '#43201b', 16, 3, false);
  }),
  silo: canvasTex(128, (g, s) => {
    g.fillStyle = '#b8bcc2'; g.fillRect(0, 0, s, s);
    lines(g, s, '#9aa0a8', 16, 3, true);
    lines(g, s, '#a6abb2', 42, 2, false);
  }),
  medkit: canvasTex(64, (g, s) => {
    g.fillStyle = '#f4f4f4'; g.fillRect(0, 0, s, s);
    g.fillStyle = '#e63946'; g.fillRect(s * 0.38, s * 0.15, s * 0.24, s * 0.7); g.fillRect(s * 0.15, s * 0.38, s * 0.7, s * 0.24);
  }),
};
TEX.post = TEX.plank;
const flashTex = canvasTex(64, (g, s) => {
  const gr = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  gr.addColorStop(0, 'rgba(255,255,230,1)'); gr.addColorStop(0.3, 'rgba(255,210,90,0.9)'); gr.addColorStop(1, 'rgba(255,120,0,0)');
  g.fillStyle = gr; g.beginPath();
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2, r = i % 2 ? s * 0.2 : s * 0.5;
    g.lineTo(s / 2 + Math.cos(a) * r, s / 2 + Math.sin(a) * r);
  }
  g.fill();
});

const TEX_SCALE = { crate: 1.2, hay: 1.1, barn: 2.5, plank: 1.5, post: 1, stone: 1.5, wall: 3, roof: 3, silo: 4 };
const WORLD_MATS = {};
for (const k of Object.keys(TEX_SCALE)) WORLD_MATS[k] = new THREE.MeshStandardMaterial({ map: TEX[k], roughness: 0.92 });

const M = (color, roughness = 0.8, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
const MAT = {
  orange: M(0xf39c12, 0.6), red: M(0xd62828, 0.7), black: M(0x111111, 0.3), white: M(0xffffff, 0.3),
  metal: M(0x4a4d52, 0.45, 0.35), wood: M(0x7a4a22, 0.8), mag: M(0x4a3a20, 0.6), egg: M(0xfff3d6, 0.5),
  trunk: M(0x6b4423), leaves: M(0x3f7d2c), leaves2: M(0x4f9238), cloud: M(0xffffff, 1),
  medkit: new THREE.MeshStandardMaterial({ map: TEX.medkit, roughness: 0.5 }),
};
const SPH = new THREE.SphereGeometry(1, 16, 12);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 10);
const CONE = new THREE.ConeGeometry(1, 1, 10);
const BOX = new THREE.BoxGeometry(1, 1, 1);

function mesh(geo, mat, x = 0, y = 0, z = 0, sx = 1, sy = sx, sz = sx) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.scale.set(sx, sy, sz);
  m.castShadow = true;
  return m;
}

// =====================================================================
// World
// =====================================================================
function buildWorld() {
  const size = ARENA * 2 + 2;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ map: TEX.grass, roughness: 1 }));
  ground.material.map.repeat.set(size / 4, size / 4);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const outerTex = TEX.grass.clone();
  outerTex.repeat.set(120, 120);
  const outer = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshStandardMaterial({ map: outerTex, color: 0xb8d8a0, roughness: 1 }));
  outer.rotation.x = -Math.PI / 2;
  outer.position.y = -0.02;
  scene.add(outer);

  for (const b of BOXES) {
    if (b.type === 'silo') continue;
    const w = b.max[0] - b.min[0], h = b.max[1] - b.min[1], d = b.max[2] - b.min[2];
    const geo = new THREE.BoxGeometry(w, h, d);
    const s = TEX_SCALE[b.type] || 2;
    const uv = geo.attributes.uv;
    const dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
    for (let f = 0; f < 6; f++) for (let k = 0; k < 4; k++) {
      const i = f * 4 + k;
      uv.setXY(i, (uv.getX(i) * dims[f][0]) / s, (uv.getY(i) * dims[f][1]) / s);
    }
    const m = new THREE.Mesh(geo, WORLD_MATS[b.type] || WORLD_MATS.crate);
    m.position.set((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    m.castShadow = m.receiveShadow = true;
    scene.add(m);
  }

  // Barn gable roof (visual only)
  const slope = Math.PI / 7, half = 8.6, slabW = half / Math.cos(slope);
  for (const s of [-1, 1]) {
    const slab = new THREE.Mesh(new THREE.BoxGeometry(slabW, 0.25, 13.6), WORLD_MATS.roof);
    slab.position.set(s * half / 2, 5.4 + (half / 2) * Math.tan(slope), 0);
    slab.rotation.z = -s * slope;
    slab.castShadow = true;
    scene.add(slab);
  }
  const tri = new THREE.Shape();
  tri.moveTo(-8.4, 0); tri.lineTo(8.4, 0); tri.lineTo(0, half * Math.tan(slope)); tri.closePath();
  const triGeo = new THREE.ShapeGeometry(tri);
  const gableMat = new THREE.MeshStandardMaterial({ color: 0xa8322d, side: THREE.DoubleSide, roughness: 0.9 });
  for (const z of [-6.3, 6.3]) {
    const g = new THREE.Mesh(triGeo, gableMat);
    g.position.set(0, 5.4, z);
    scene.add(g);
  }

  for (const [x, z] of SILOS) {
    const body = new THREE.Mesh(new THREE.CylinderGeometry(2, 2, 9, 24), new THREE.MeshStandardMaterial({ map: TEX.silo, roughness: 0.5, metalness: 0.3 }));
    body.material.map.repeat.set(3, 1);
    body.position.set(x, 4.5, z);
    body.castShadow = body.receiveShadow = true;
    scene.add(body);
    const dome = new THREE.Mesh(new THREE.SphereGeometry(2, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), M(0x8a3a32, 0.6, 0.2));
    dome.position.set(x, 9, z);
    dome.castShadow = true;
    scene.add(dome);
  }

  // Scenery outside the fence
  for (let i = 0; i < 70; i++) {
    const a = Math.random() * Math.PI * 2, r = rnd(46, 110);
    const x = Math.cos(a) * r, z = Math.sin(a) * r, s = rnd(0.8, 1.6);
    scene.add(mesh(CYL, MAT.trunk, x, 1.2 * s, z, 0.3 * s, 2.4 * s, 0.3 * s));
    scene.add(mesh(CONE, i % 2 ? MAT.leaves : MAT.leaves2, x, 3.6 * s, z, 1.8 * s, 3.4 * s, 1.8 * s));
    scene.add(mesh(CONE, i % 2 ? MAT.leaves2 : MAT.leaves, x, 5.1 * s, z, 1.3 * s, 2.6 * s, 1.3 * s));
  }
  for (let i = 0; i < 14; i++) {
    const c = new THREE.Group();
    for (let k = 0; k < 4; k++) c.add(mesh(SPH, MAT.cloud, k * 3 - 4.5, rnd(-0.5, 0.8), rnd(-1.5, 1.5), rnd(2.5, 4), rnd(1.5, 2.2), rnd(2.5, 3.5)));
    c.children.forEach((m) => (m.castShadow = false));
    const a = Math.random() * Math.PI * 2, r = rnd(30, 140);
    c.position.set(Math.cos(a) * r, rnd(45, 65), Math.sin(a) * r);
    scene.add(c);
  }
}
buildWorld();

// Pickups (medkit + egg refill)
const pickupMeshes = PICKUPS.map((p) => {
  const g = new THREE.Group();
  g.add(mesh(BOX, MAT.medkit, 0, 0, 0, 0.55, 0.38, 0.38));
  g.add(mesh(SPH, MAT.egg, 0.42, 0, 0, 0.12, 0.16, 0.12));
  g.position.set(p[0], p[1] + 0.6, p[2]);
  scene.add(g);
  return g;
});

// =====================================================================
// Chickens & guns
// =====================================================================
function makeGun(i) {
  const g = new THREE.Group();
  const add = (geo, mat, x, y, z, sx, sy, sz, rx = 0) => { const m = mesh(geo, mat, x, y, z, sx, sy, sz); m.rotation.x = rx; g.add(m); return m; };
  const R90 = Math.PI / 2;
  switch (i) {
    case 0: // pistol
      add(BOX, MAT.metal, 0, 0.02, -0.08, 0.07, 0.09, 0.32);
      add(CYL, MAT.metal, 0, 0.03, -0.26, 0.025, 0.08, 0.025, R90);
      add(BOX, MAT.wood, 0, -0.09, 0.03, 0.06, 0.16, 0.08, 0.25);
      break;
    case 1: // double-barrel shotgun
      add(CYL, MAT.metal, -0.033, 0.03, -0.32, 0.032, 0.62, 0.032, R90);
      add(CYL, MAT.metal, 0.033, 0.03, -0.32, 0.032, 0.62, 0.032, R90);
      add(BOX, MAT.wood, 0, -0.03, 0.12, 0.09, 0.13, 0.42);
      add(BOX, MAT.wood, 0, -0.03, -0.22, 0.1, 0.06, 0.22);
      break;
    case 2: // rifle
      add(BOX, MAT.metal, 0, 0.02, -0.05, 0.08, 0.11, 0.45);
      add(CYL, MAT.metal, 0, 0.04, -0.42, 0.022, 0.32, 0.022, R90);
      add(BOX, MAT.mag, 0, -0.12, -0.12, 0.06, 0.2, 0.09, -0.3);
      add(BOX, MAT.wood, 0, -0.02, 0.28, 0.07, 0.11, 0.24);
      add(BOX, MAT.wood, 0, -0.01, -0.3, 0.075, 0.07, 0.16);
      break;
    case 3: // sniper
      add(BOX, MAT.wood, 0, -0.01, 0.06, 0.08, 0.11, 0.62);
      add(BOX, MAT.metal, 0, 0.03, -0.15, 0.07, 0.08, 0.3);
      add(CYL, MAT.metal, 0, 0.03, -0.6, 0.022, 0.62, 0.022, R90);
      add(CYL, MAT.black, 0, 0.12, -0.08, 0.042, 0.34, 0.042, R90);
      add(BOX, MAT.metal, 0, 0.08, -0.08, 0.02, 0.05, 0.05);
      break;
  }
  return g;
}

function makeFlashSprite(scale) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: flashTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.setScalar(scale);
  s.visible = false;
  return s;
}

function makeChicken(color) {
  const group = new THREE.Group();
  const body = M(color, 0.85);
  group.add(mesh(SPH, body, 0, 0.6, 0, 0.43, 0.45, 0.52));
  for (const [x, rz] of [[-0.08, 0.35], [0, 0], [0.08, -0.35]]) {
    const t = mesh(SPH, body, x, 0.95, 0.42, 0.07, 0.22, 0.1);
    t.rotation.set(-0.5, 0, rz);
    group.add(t);
  }
  const head = new THREE.Group();
  head.position.set(0, 1.05, -0.28);
  group.add(head);
  head.add(mesh(SPH, body, 0, 0, 0, 0.27));
  const beak = mesh(CONE, MAT.orange, 0, -0.03, -0.3, 0.08, 0.2, 0.08);
  beak.rotation.x = -Math.PI / 2;
  head.add(beak);
  head.add(mesh(SPH, MAT.red, 0, -0.15, -0.22, 0.05, 0.09, 0.05));
  for (const [y, z] of [[0.25, -0.1], [0.29, 0], [0.26, 0.1]]) head.add(mesh(SPH, MAT.red, 0, y, z, 0.06, 0.08, 0.06));
  for (const s of [-1, 1]) {
    head.add(mesh(SPH, MAT.white, s * 0.15, 0.06, -0.19, 0.06));
    head.add(mesh(SPH, MAT.black, s * 0.168, 0.065, -0.23, 0.035));
  }
  const legs = [], wings = [];
  for (const s of [-1, 1]) {
    const leg = new THREE.Group();
    leg.position.set(s * 0.17, 0.32, 0);
    leg.add(mesh(CYL, MAT.orange, 0, -0.16, 0, 0.035, 0.32, 0.035));
    leg.add(mesh(BOX, MAT.orange, 0, -0.31, -0.06, 0.16, 0.04, 0.24));
    group.add(leg); legs.push(leg);
    const wing = new THREE.Group();
    wing.position.set(s * 0.4, 0.8, 0);
    wing.add(mesh(SPH, body, 0, -0.18, 0.05, 0.08, 0.24, 0.32));
    group.add(wing); wings.push(wing);
  }
  const gunHolder = new THREE.Group();
  gunHolder.position.set(0.3, 0.72, -0.28);
  group.add(gunHolder);
  const guns = WEAPONS.map((_, i) => { const g = makeGun(i); g.visible = false; gunHolder.add(g); return g; });
  const flash = makeFlashSprite(0.5);
  gunHolder.add(flash);
  const ch = {
    group, head, legs, wings, gunHolder, guns, flash, w: -1, walk: 0, flashUntil: 0,
    setWeapon(i) {
      if (ch.w === i || !WEAPONS[i]) return;
      ch.w = i;
      guns.forEach((g, k) => (g.visible = k === i));
      flash.position.set(0, 0.03, WEAPONS[i].muzzle - 0.1);
    },
  };
  ch.setWeapon(0);
  return ch;
}

function animateChicken(ch, hSpeed, vy, dt, t) {
  ch.walk += dt * Math.min(hSpeed, 10) * 2.4;
  const amp = Math.min(1, hSpeed / 5) * 0.7;
  ch.legs[0].rotation.x = Math.sin(ch.walk) * amp;
  ch.legs[1].rotation.x = -Math.sin(ch.walk) * amp;
  const flap = Math.abs(vy) > 0.8 ? Math.abs(Math.sin(t * 22)) * 0.9 : 0;
  ch.wings[0].rotation.z = -flap;
  ch.wings[1].rotation.z = flap * 0.3;
  ch.flash.visible = t < ch.flashUntil;
  if (ch.flash.visible) ch.flash.material.rotation = Math.random() * 6;
}

function makeNameTag(name) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.font = 'bold 34px Segoe UI, Arial';
  const w = Math.min(250, g.measureText(name).width + 24);
  g.fillStyle = 'rgba(0,0,0,0.45)';
  g.beginPath(); g.roundRect(128 - w / 2, 8, w, 48, 12); g.fill();
  g.fillStyle = '#fff'; g.textAlign = 'center'; g.fillText(name, 128, 44);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false }));
  s.scale.set(1.6, 0.4, 1);
  s.position.y = 1.75;
  return s;
}

// =====================================================================
// Viewmodel (first-person gun + wing)
// =====================================================================
const vm = new THREE.Group();
vmScene.add(vm);
const vmGuns = WEAPONS.map((_, i) => { const g = makeGun(i); g.visible = false; vm.add(g); return g; });
const vmWingMat = M(settings.color, 0.85);
const vmWing = mesh(SPH, vmWingMat, 0.02, -0.13, 0.12, 0.1, 0.13, 0.3);
vm.add(vmWing);
const vmFlash = makeFlashSprite(0.4);
vm.add(vmFlash);
const vmState = { kick: 0, swayX: 0, swayY: 0, flashUntil: 0 };

// =====================================================================
// Audio (all synthesized)
// =====================================================================
let actx = null, master = null, noiseBuf = null;
function initAudio() {
  if (actx) { actx.resume(); return; }
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    master = actx.createGain();
    master.gain.value = settings.vol;
    master.connect(actx.destination);
    noiseBuf = actx.createBuffer(1, actx.sampleRate, actx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  } catch { actx = null; }
}
const camFwd = new THREE.Vector3(0, 0, -1);
function outNode(pos, base = 1) {
  if (!actx) return null;
  let vol = base, pan = 0;
  if (pos) {
    const dx = pos[0] - camera.position.x, dy = pos[1] - camera.position.y, dz = pos[2] - camera.position.z;
    const d = Math.hypot(dx, dy, dz);
    vol = base / (1 + d * 0.09);
    if (d > 0.5) pan = Math.max(-1, Math.min(1, (dx * -camFwd.z + dz * camFwd.x) / d)) * 0.8;
  }
  if (vol < 0.01) return null;
  const g = actx.createGain();
  g.gain.value = vol;
  const p = actx.createStereoPanner();
  p.pan.value = pan;
  g.connect(p).connect(master);
  return g;
}
function env(g, t, peak, dur, attack = 0.005) {
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
}
function noise(dest, t, dur, freq, type = 'lowpass', peak = 1, q = 0.7) {
  const s = actx.createBufferSource(); s.buffer = noiseBuf;
  const f = actx.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q;
  const g = actx.createGain(); env(g, t, peak, dur);
  s.connect(f).connect(g).connect(dest);
  s.start(t, Math.random() * 0.4); s.stop(t + dur + 0.05);
}
function tone(dest, t, dur, f0, f1, type = 'sine', peak = 1) {
  const o = actx.createOscillator(); o.type = type;
  o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  const g = actx.createGain(); env(g, t, peak, dur);
  o.connect(g).connect(dest);
  o.start(t); o.stop(t + dur + 0.05);
}
const SHOT_SND = [
  { f: 2600, d: 0.18, b: 160, v: 0.55 }, { f: 1300, d: 0.4, b: 90, v: 1 },
  { f: 3000, d: 0.13, b: 140, v: 0.45 }, { f: 1700, d: 0.7, b: 70, v: 1 },
];
const snd = {
  shot(w, pos) {
    const s = SHOT_SND[w], o = outNode(pos, s.v); if (!o) return;
    const t = actx.currentTime;
    noise(o, t, s.d, s.f, 'lowpass', 1);
    tone(o, t, s.d * 0.7, s.b, 35, 'sine', 1.2);
  },
  bawk(pos, big) {
    const o = outNode(pos, big ? 0.7 : 0.4); if (!o) return;
    const notes = big
      ? [[520, 780, 0.09], [0, 0, 0.03], [620, 900, 0.08], [0, 0, 0.02], [950, 420, 0.3]]
      : [[600 + rnd(-80, 80), 850, 0.07], [0, 0, 0.03], [820, 480, 0.15]];
    let t = actx.currentTime;
    for (const [a, b, d] of notes) {
      if (a) {
        const osc = actx.createOscillator(); osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(a, t); osc.frequency.exponentialRampToValueAtTime(b, t + d);
        const f = actx.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 1500; f.Q.value = 1.2;
        const g = actx.createGain(); env(g, t, 0.9, d, 0.01);
        osc.connect(f).connect(g).connect(o);
        osc.start(t); osc.stop(t + d + 0.05);
      }
      t += d;
    }
  },
  boom(pos) {
    const o = outNode(pos, 1.4); if (!o) return;
    const t = actx.currentTime;
    noise(o, t, 1.3, 500, 'lowpass', 1.5);
    tone(o, t, 0.9, 90, 28, 'sine', 1.5);
  },
  hit(hs) { const o = outNode(null, 0.25); if (o) tone(o, actx.currentTime, 0.06, hs ? 1900 : 1200, hs ? 1700 : 1000, 'square', 0.5); },
  kill() {
    const o = outNode(null, 0.3); if (!o) return;
    const t = actx.currentTime;
    tone(o, t, 0.08, 900, 900, 'square', 0.5); tone(o, t + 0.09, 0.15, 1350, 1350, 'square', 0.5);
  },
  click() { const o = outNode(null, 0.3); if (o) tone(o, actx.currentTime, 0.03, 900, 700, 'square', 0.4); },
  reload(dur) {
    const o = outNode(null, 0.35); if (!o) return;
    const t = actx.currentTime;
    noise(o, t + 0.1, 0.06, 3000, 'bandpass', 1, 2);
    noise(o, t + dur - 0.2, 0.08, 2200, 'bandpass', 1, 2);
  },
  pickup() {
    const o = outNode(null, 0.3); if (!o) return;
    const t = actx.currentTime;
    [600, 900, 1200].forEach((f, i) => tone(o, t + i * 0.07, 0.12, f, f, 'triangle', 0.7));
  },
  whoosh() { const o = outNode(null, 0.3); if (o) noise(o, actx.currentTime, 0.25, 900, 'bandpass', 0.8, 1); },
};

// =====================================================================
// Effects: particles, tracers, explosions
// =====================================================================
const particles = [];
const PART_GEO = { feather: new THREE.PlaneGeometry(0.17, 0.06), cube: new THREE.BoxGeometry(0.08, 0.08, 0.08) };
const partMats = new Map();
function partMat(color) {
  if (!partMats.has(color)) partMats.set(color, new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide }));
  return partMats.get(color);
}
function spawnParticle(pos, vel, color, life, kind, grav, drag) {
  if (particles.length > 600) { const old = particles.shift(); scene.remove(old.m); }
  const m = new THREE.Mesh(PART_GEO[kind], partMat(color));
  m.position.copy(pos);
  m.rotation.set(rnd(0, 6), rnd(0, 6), rnd(0, 6));
  scene.add(m);
  particles.push({ m, vel, life, grav, drag, spin: new THREE.Vector3(rnd(-8, 8), rnd(-8, 8), rnd(-8, 8)) });
}
function featherBurst(pos, color, n) {
  for (let i = 0; i < n; i++) {
    const v = new THREE.Vector3(rnd(-1, 1), rnd(0.2, 1.4), rnd(-1, 1)).multiplyScalar(rnd(2, 5));
    spawnParticle(pos, v, color, rnd(1.2, 2.6), 'feather', 2.5, 2.2);
  }
}
function dustPuff(pos, n = 5) {
  for (let i = 0; i < n; i++) {
    spawnParticle(pos, new THREE.Vector3(rnd(-1.5, 1.5), rnd(0.5, 2.5), rnd(-1.5, 1.5)), 0x9a8a70, rnd(0.3, 0.6), 'cube', 6, 1);
  }
}
function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= dt;
    if (p.life <= 0) { scene.remove(p.m); particles.splice(i, 1); continue; }
    p.vel.y -= p.grav * dt;
    p.vel.multiplyScalar(Math.max(0, 1 - p.drag * dt));
    p.m.position.addScaledVector(p.vel, dt);
    if (p.m.position.y < 0.03) { p.m.position.y = 0.03; p.vel.set(0, 0, 0); p.spin.set(0, 0, 0); }
    p.m.rotation.x += p.spin.x * dt; p.m.rotation.y += p.spin.y * dt; p.m.rotation.z += p.spin.z * dt;
    p.m.scale.setScalar(Math.min(1, p.life / 0.3));
  }
}

const tracerGeo = new THREE.BoxGeometry(0.02, 0.02, 1).translate(0, 0, 0.5);
const tracerMat = new THREE.MeshBasicMaterial({ color: 0xffe9a0, transparent: true, opacity: 0.8 });
const tracers = [];
function addTracer(start, end) {
  const len = start.distanceTo(end);
  if (len < 0.5) return;
  const m = new THREE.Mesh(tracerGeo, tracerMat);
  m.position.copy(start);
  m.lookAt(end);
  m.scale.z = len;
  scene.add(m);
  tracers.push({ m, life: 0.06 });
}

const booms = [];
let shake = 0;
function explosion(p) {
  const pos = new THREE.Vector3(...p);
  const m = new THREE.Mesh(SPH, new THREE.MeshBasicMaterial({ color: 0xffaa33, transparent: true, opacity: 0.9, depthWrite: false }));
  m.position.copy(pos);
  scene.add(m);
  booms.push({ m, t: 0 });
  for (let i = 0; i < 30; i++) {
    const v = new THREE.Vector3(rnd(-1, 1), rnd(0.3, 1.5), rnd(-1, 1)).multiplyScalar(rnd(3, 9));
    spawnParticle(pos, v, i % 3 ? 0xffc21a : 0xff8c00, rnd(0.5, 1.2), 'cube', 14, 0.5);
  }
  for (let i = 0; i < 16; i++) {
    const v = new THREE.Vector3(rnd(-1, 1), rnd(0.5, 1.5), rnd(-1, 1)).multiplyScalar(rnd(3, 7));
    spawnParticle(pos, v, 0xfff8e7, rnd(0.8, 1.5), 'feather', 10, 0.5);
  }
  lightFlash(pos, 600, 0.35, 25);
  snd.boom(p);
  const d = camera.position.distanceTo(pos);
  if (d < 16) shake = Math.max(shake, 1 - d / 16);
}
function updateEffects(dt) {
  for (let i = tracers.length - 1; i >= 0; i--) {
    if ((tracers[i].life -= dt) <= 0) { scene.remove(tracers[i].m); tracers.splice(i, 1); }
  }
  for (let i = booms.length - 1; i >= 0; i--) {
    const b = booms[i];
    b.t += dt;
    const k = b.t / 0.4;
    if (k >= 1) { scene.remove(b.m); b.m.material.dispose(); booms.splice(i, 1); continue; }
    b.m.scale.setScalar(0.5 + k * EGG.radius * 0.8);
    b.m.material.opacity = 0.9 * (1 - k);
  }
  flashState.t += dt;
  flashLight.intensity = flashState.t < flashState.dur ? flashState.peak * (1 - flashState.t / flashState.dur) : 0;
  shake = Math.max(0, shake - dt * 2.5);
}

// =====================================================================
// Game state
// =====================================================================
let ws = null, myId = 0, joined = false, matchOver = false, scoreLimit = SCORE_LIMIT;
const info = new Map();    // id -> { name, color, kills, deaths }
const remotes = new Map(); // id -> remote chicken state
const eggMeshes = new Map();
const me = {
  pos: new THREE.Vector3(), vel: new THREE.Vector3(), yaw: 0, pitch: 0, viewY: 0,
  onGround: false, hp: 100, alive: false, w: 0, ammo: WEAPONS.map((w) => w.mag),
  reloadUntil: 0, nextShot: 0, switchUntil: 0, eggs: EGG.max, nextEgg: 0, aiming: false,
  deathTime: 0, triggerUsed: false, bob: 0,
};
const deathPos = new THREE.Vector3();
let thirdPerson = false;
let myChicken = null;
let lastSend = 0, damageFlash = 0;
const lastBawk = new Map();

// =====================================================================
// Input
// =====================================================================
const keys = {};
let mouseDown = false, locked = false, chatOpen = false;

function lock() {
  try { const p = canvas.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch { /* ignore */ }
}
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === canvas;
  if (locked) {
    $('menu').classList.add('hidden');
  } else if (joined) {
    if (chatOpen) closeChat();
    showMenu();
    mouseDown = false; me.aiming = false;
    for (const k in keys) keys[k] = false;
  }
});
canvas.addEventListener('click', () => { if (joined && !locked) lock(); });

document.addEventListener('keydown', (e) => {
  if (chatOpen || !joined) return;
  if (['Tab', 'Space'].includes(e.code)) e.preventDefault();
  keys[e.code] = true;
  if (e.repeat) return;
  const now = nowSec();
  if (e.code >= 'Digit1' && e.code <= 'Digit4') switchWeapon(+e.code.slice(5) - 1, now);
  else if (e.code === 'KeyR') startReload(now);
  else if (e.code === 'KeyG') throwEgg(now);
  else if (e.code === 'KeyV') thirdPerson = !thirdPerson;
  else if ((e.code === 'KeyT' || e.code === 'Enter') && locked) { e.preventDefault(); openChat(); }
});
document.addEventListener('keyup', (e) => { keys[e.code] = false; });
document.addEventListener('mousemove', (e) => {
  if (!locked || !me.alive) return;
  const zoomFactor = me.aiming ? camera.fov / BASE_FOV : 1;
  const k = 0.0022 * settings.sens * zoomFactor;
  me.yaw -= e.movementX * k;
  me.pitch = Math.max(-1.5, Math.min(1.5, me.pitch - e.movementY * k));
  vmState.swayX = Math.max(-1, Math.min(1, vmState.swayX + e.movementX * 0.002));
  vmState.swayY = Math.max(-1, Math.min(1, vmState.swayY + e.movementY * 0.002));
});
document.addEventListener('mousedown', (e) => {
  if (!locked) return;
  if (e.button === 0) { mouseDown = true; me.triggerUsed = false; }
  if (e.button === 2) me.aiming = true;
});
document.addEventListener('mouseup', (e) => {
  if (e.button === 0) mouseDown = false;
  if (e.button === 2) me.aiming = false;
});
document.addEventListener('contextmenu', (e) => e.preventDefault());
document.addEventListener('wheel', (e) => {
  if (!locked) return;
  const n = WEAPONS.length;
  switchWeapon((me.w + (e.deltaY > 0 ? 1 : n - 1)) % n, nowSec());
}, { passive: true });

// Chat
const chatInput = $('chatInput');
function openChat() {
  chatOpen = true;
  for (const k in keys) keys[k] = false;
  mouseDown = false;
  chatInput.classList.remove('hidden');
  chatInput.value = '';
  setTimeout(() => chatInput.focus(), 0);
  document.querySelectorAll('#chatLog div').forEach((d) => (d.style.opacity = 1));
}
function closeChat() {
  chatOpen = false;
  chatInput.blur();
  chatInput.classList.add('hidden');
}
chatInput.addEventListener('keydown', (e) => {
  e.stopPropagation();
  if (e.key === 'Enter') {
    const text = chatInput.value.trim();
    if (text) send({ t: 'chat', m: text });
    closeChat();
  } else if (e.key === 'Escape') closeChat();
});
function addChat(html) {
  const log = $('chatLog');
  const d = document.createElement('div');
  d.innerHTML = html;
  log.appendChild(d);
  while (log.children.length > 8) log.firstChild.remove();
  setTimeout(() => { if (!chatOpen) d.style.opacity = 0; }, 12000);
}
const nameHTML = (id) => {
  const p = info.get(id);
  return p ? `<b style="color:${p.color === '#333333' ? '#aaa' : p.color}">${esc(p.name)}</b>` : '<b>???</b>';
};

// =====================================================================
// Local player: movement, collision, weapons
// =====================================================================
const { R, H, EYE, STEP } = PLAYER;
function overlapBox(x, y, z, b) {
  return x + R > b.min[0] && x - R < b.max[0] && z + R > b.min[2] && z - R < b.max[2] && y + H > b.min[1] && y < b.max[1];
}
function overlapsAny(x, y, z) {
  for (const b of BOXES) if (overlapBox(x, y, z, b)) return true;
  return false;
}
function moveAxis(axis, delta) {
  if (!delta) return;
  const p = me.pos, key = axis === 0 ? 'x' : 'z';
  p[key] += delta;
  for (const b of BOXES) {
    if (!overlapBox(p.x, p.y, p.z, b)) continue;
    const rise = b.max[1] - p.y;
    if (rise > 0 && rise <= STEP && (me.onGround || me.vel.y <= 0) && !overlapsAny(p.x, b.max[1] + 0.001, p.z)) {
      p.y = b.max[1];
      continue;
    }
    p[key] = delta > 0 ? b.min[axis] - R - 0.001 : b.max[axis] + R + 0.001;
    me.vel[key] = 0;
  }
}
function moveVertical(dy) {
  const p = me.pos;
  p.y += dy;
  me.onGround = false;
  if (p.y <= 0) { p.y = 0; me.vel.y = 0; me.onGround = true; }
  for (const b of BOXES) {
    if (!overlapBox(p.x, p.y, p.z, b)) continue;
    if (dy <= 0) { p.y = b.max[1]; me.onGround = true; } else { p.y = b.min[1] - H - 0.001; }
    me.vel.y = 0;
  }
}

function updateLocal(dt, now) {
  const fx = -Math.sin(me.yaw), fz = -Math.cos(me.yaw);
  let f = 0, s = 0;
  if (!chatOpen && locked) {
    f = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0);
    s = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0);
  }
  let wx = fx * f - fz * s, wz = fz * f + fx * s;
  const wl = Math.hypot(wx, wz);
  if (wl > 0) { wx /= wl; wz /= wl; }
  const sprint = keys.ShiftLeft && f > 0 && !me.aiming;
  const speed = (sprint ? 9.5 : 6.5) * (me.aiming ? 0.55 : 1);
  const accel = Math.min(1, (me.onGround ? 14 : 3) * dt);
  me.vel.x += (wx * speed - me.vel.x) * accel;
  me.vel.z += (wz * speed - me.vel.z) * accel;
  if (keys.Space && me.onGround) { me.vel.y = 8.6; me.onGround = false; }
  me.vel.y -= 24 * dt;

  const steps = Math.max(1, Math.ceil((me.vel.length() * dt) / 0.15));
  const sdt = dt / steps;
  for (let i = 0; i < steps; i++) {
    moveAxis(0, me.vel.x * sdt);
    moveAxis(2, me.vel.z * sdt);
    moveVertical(me.vel.y * sdt);
  }

  const W = WEAPONS[me.w];
  if (me.reloadUntil && now >= me.reloadUntil) { me.ammo[me.w] = W.mag; me.reloadUntil = 0; }
  if (mouseDown && locked && !chatOpen && (W.auto || !me.triggerUsed)) {
    if (tryShoot(now)) me.triggerUsed = true;
  }

  const hs = Math.hypot(me.vel.x, me.vel.z);
  if (me.onGround && hs > 0.5) me.bob += dt * hs * 1.8;
  if (myChicken) animateChicken(myChicken, hs, me.onGround ? 0 : me.vel.y, dt, now);
}

function eyePos() { return new THREE.Vector3(me.pos.x, me.pos.y + EYE, me.pos.z); }
function aimForward() {
  const cp = Math.cos(me.pitch);
  return new THREE.Vector3(-Math.sin(me.yaw) * cp, Math.sin(me.pitch), -Math.cos(me.yaw) * cp);
}
function aimRaycast(o, d, max) {
  const oa = o.toArray(), da = d.toArray();
  let t = raycastWorld(oa, da, max), hit = false;
  for (const r of remotes.values()) {
    if (!r.alive) continue;
    const h = chickenHit(oa, da, r.pos.toArray(), r.ry);
    if (h.t < t) { t = h.t; hit = true; }
  }
  return { t, hit };
}
function currentSpread() {
  const W = WEAPONS[me.w];
  let spread = W.spread;
  if (me.w === 3) spread = me.aiming ? W.scopedSpread : W.spread;
  else if (me.aiming) spread *= me.w === 1 ? 0.8 : 0.45;
  if (Math.hypot(me.vel.x, me.vel.z) > 1) spread *= 1.5;
  if (!me.onGround) spread *= 2;
  return spread;
}
function muzzleWorld() {
  const W = WEAPONS[me.w];
  if (thirdPerson && myChicken) {
    myChicken.group.updateMatrixWorld(true);
    return myChicken.gunHolder.localToWorld(new THREE.Vector3(0, 0.03, W.muzzle));
  }
  const fwd = aimForward(), right = new THREE.Vector3(Math.cos(me.yaw), 0, -Math.sin(me.yaw));
  const up = new THREE.Vector3().crossVectors(right, fwd);
  return camera.position.clone().addScaledVector(fwd, 0.7).addScaledVector(right, 0.2).addScaledVector(up, -0.16);
}

function tryShoot(now) {
  const W = WEAPONS[me.w];
  if (!me.alive || now < me.nextShot || now < me.switchUntil || me.reloadUntil) return false;
  if (me.ammo[me.w] <= 0) { snd.click(); startReload(now); return true; }
  me.ammo[me.w]--;
  me.nextShot = now + W.rate;

  const eye = eyePos();
  const camDir = aimForward();
  const aim = aimRaycast(camera.position, camDir, 300);
  const aimPoint = camera.position.clone().addScaledVector(camDir, aim.t);
  const base = aimPoint.sub(eye);
  if (base.lengthSq() < 0.25) base.copy(camDir); else base.normalize();
  const right = new THREE.Vector3().crossVectors(base, new THREE.Vector3(0, 1, 0)).normalize();
  const up = new THREE.Vector3().crossVectors(right, base);
  const spread = currentSpread();
  const dirs = [];
  for (let i = 0; i < W.pellets; i++) {
    const r = spread * Math.sqrt(Math.random()), a = Math.random() * Math.PI * 2;
    dirs.push(base.clone().addScaledVector(right, r * Math.cos(a)).addScaledVector(up, r * Math.sin(a)).normalize());
  }
  send({ t: 'shoot', w: me.w, o: eye.toArray().map((v) => +v.toFixed(3)), d: dirs.map((d) => d.toArray().map((v) => +v.toFixed(4))) });

  const start = muzzleWorld();
  for (const d of dirs) {
    const h = aimRaycast(eye, d, W.range);
    const end = eye.clone().addScaledVector(d, h.t);
    addTracer(start, end);
    if (!h.hit && h.t < W.range) dustPuff(end, 4);
  }
  snd.shot(me.w, null);
  lightFlash(start, 25, 0.06, 8);
  vmState.flashUntil = now + 0.05;
  if (myChicken) myChicken.flashUntil = now + 0.05;
  vmState.kick = 1;
  me.pitch = Math.min(1.5, me.pitch + W.recoil * rnd(0.8, 1.2));
  me.yaw += rnd(-0.5, 0.5) * W.recoil * 0.6;
  if (me.ammo[me.w] === 0) me.autoReloadAt = now + W.rate;
  return true;
}
function startReload(now) {
  const W = WEAPONS[me.w];
  if (!me.alive || me.reloadUntil || me.ammo[me.w] >= W.mag) return;
  me.reloadUntil = now + W.reload;
  snd.reload(W.reload);
}
function switchWeapon(i, now) {
  if (i === me.w || !WEAPONS[i]) return;
  me.w = i;
  me.reloadUntil = 0;
  me.autoReloadAt = 0;
  me.switchUntil = now + 0.35;
  me.triggerUsed = true;
  if (myChicken) myChicken.setWeapon(i);
}
function throwEgg(now) {
  if (!me.alive || me.eggs <= 0 || now < me.nextEgg || matchOver) return;
  me.nextEgg = now + 0.8;
  me.eggs--;
  const d = aimForward();
  const o = eyePos().addScaledVector(d, 0.6);
  const v = d.clone().multiplyScalar(EGG.speed).add(new THREE.Vector3(me.vel.x * 0.4, 3, me.vel.z * 0.4));
  send({ t: 'egg', o: o.toArray(), v: v.toArray() });
  snd.whoosh();
}

function die(killerId) {
  me.alive = false;
  me.aiming = false;
  me.deathTime = nowSec();
  deathPos.copy(me.pos);
  featherBurst(deathPos.clone().setY(deathPos.y + 0.6), settings.color, 40);
  snd.bawk(null, true);
  let by = 'Fried!';
  if (killerId === myId) by = 'You fried yourself 🤦';
  else if (info.has(killerId)) by = `Fried by ${info.get(killerId).name}`;
  setText($('deathBy'), by);
  $('death').classList.remove('hidden');
}
function respawnLocal(m) {
  me.pos.set(...m.p);
  me.viewY = me.pos.y;
  me.vel.set(0, 0, 0);
  me.yaw = m.ry; me.pitch = 0;
  me.hp = m.hp; me.eggs = m.eggs;
  me.ammo = WEAPONS.map((w) => w.mag);
  me.reloadUntil = 0; me.autoReloadAt = 0;
  me.alive = true;
  $('death').classList.add('hidden');
}

// =====================================================================
// Networking
// =====================================================================
function send(msg) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); }

function connect() {
  setText($('status'), 'Connecting…');
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`);
  ws.onopen = () => send({ t: 'join', name: settings.name, color: settings.color });
  ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => {
    joined = false; ws = null; me.alive = false;
    $('hud').classList.add('hidden');
    $('joinFields').classList.remove('hidden');
    setText($('play'), 'RECONNECT');
    setText($('status'), 'Disconnected from server.');
    $('menu').classList.remove('hidden');
    if (document.pointerLockElement) document.exitPointerLock();
    for (const id of [...remotes.keys()]) removeRemote(id);
    for (const [, m] of eggMeshes) scene.remove(m);
    eggMeshes.clear();
    info.clear();
  };
}

function createRemote(p) {
  if (remotes.has(p.id)) return;
  const ch = makeChicken(p.color);
  ch.group.add(makeNameTag(p.name));
  scene.add(ch.group);
  const pos = new THREE.Vector3(...(p.pos || [0, 0, 0]));
  remotes.set(p.id, { ch, pos, target: pos.clone(), ry: p.ry || 0, tRy: p.ry || 0, rx: 0, tRx: 0, alive: !!p.alive, prevY: pos.y, vy: 0 });
  ch.setWeapon(p.w || 0);
}
function removeRemote(id) {
  const r = remotes.get(id);
  if (r) { scene.remove(r.ch.group); remotes.delete(id); }
}
function posOf(id) {
  if (id === myId) return me.pos;
  const r = remotes.get(id);
  return r ? r.pos : null;
}
function applyScores(s) {
  for (const [id, k, d] of s) { const p = info.get(id); if (p) { p.kills = k; p.deaths = d; } }
}

function onMessage(m) {
  switch (m.t) {
    case 'welcome': {
      myId = m.id; joined = true; matchOver = m.matchOver; scoreLimit = m.scoreLimit;
      for (const p of m.players) {
        info.set(p.id, { name: p.name, color: p.color, kills: p.kills, deaths: p.deaths });
        if (p.id !== myId) createRemote(p);
      }
      m.pickups.forEach((a, i) => (pickupMeshes[i].visible = !!a));
      if (myChicken) scene.remove(myChicken.group);
      myChicken = makeChicken(settings.color);
      scene.add(myChicken.group);
      vmWingMat.color.set(settings.color);
      $('hud').classList.remove('hidden');
      $('joinFields').classList.add('hidden');
      setText($('play'), 'RESUME');
      setText($('status'), '');
      addChat('🐔 <i>Welcome to the farm! First to ' + scoreLimit + ' kills wins.</i>');
      break;
    }
    case 'join':
      info.set(m.p.id, { name: m.p.name, color: m.p.color, kills: m.p.kills, deaths: m.p.deaths });
      createRemote(m.p);
      addChat(`${nameHTML(m.p.id)} <i>joined the farm</i>`);
      break;
    case 'leave':
      addChat(`${nameHTML(m.id)} <i>left</i>`);
      removeRemote(m.id);
      info.delete(m.id);
      break;
    case 'snap': {
      for (const [id, x, y, z, ry, rx, w, alive] of m.p) {
        if (id === myId) continue;
        const r = remotes.get(id);
        if (!r) continue;
        r.target.set(x, y, z);
        r.tRy = ry; r.tRx = rx;
        r.ch.setWeapon(w);
        if (alive && !r.alive) r.pos.copy(r.target);
        r.alive = !!alive;
      }
      const seen = new Set();
      for (const [id, x, y, z] of m.e) {
        seen.add(id);
        let em = eggMeshes.get(id);
        if (!em) {
          em = mesh(SPH, MAT.egg, x, y, z, 0.12, 0.16, 0.12);
          em.userData.target = new THREE.Vector3(x, y, z);
          scene.add(em);
          eggMeshes.set(id, em);
        }
        em.userData.target.set(x, y, z);
      }
      for (const [id, em] of eggMeshes) if (!seen.has(id)) { scene.remove(em); eggMeshes.delete(id); }
      break;
    }
    case 'spawn':
      if (m.id === myId) respawnLocal(m);
      else {
        const r = remotes.get(m.id);
        if (r) { r.pos.set(...m.p); r.target.set(...m.p); r.ry = r.tRy = m.ry; r.alive = true; }
      }
      break;
    case 'shot': {
      const r = remotes.get(m.id);
      if (!r) break;
      r.ch.group.updateMatrixWorld(true);
      const start = r.ch.gunHolder.localToWorld(new THREE.Vector3(0, 0.03, WEAPONS[m.w].muzzle));
      for (const e of m.e) {
        const end = new THREE.Vector3(...e);
        addTracer(start, end);
        dustPuff(end, 3);
      }
      r.ch.flashUntil = nowSec() + 0.05;
      snd.shot(m.w, start.toArray());
      break;
    }
    case 'hit': {
      const color = info.get(m.v)?.color || '#ffffff';
      featherBurst(new THREE.Vector3(...m.p), color, m.hs ? 14 : 6);
      const t = nowSec();
      if (m.v === myId) {
        me.hp = m.hp;
        damageFlash = Math.min(1, damageFlash + m.d / 40);
        shake = Math.max(shake, Math.min(0.5, m.d / 100));
        if (t - (lastBawk.get(m.v) || 0) > 0.3) { snd.bawk(null, false); lastBawk.set(m.v, t); }
      } else if (t - (lastBawk.get(m.v) || 0) > 0.3) {
        snd.bawk(m.p, false);
        lastBawk.set(m.v, t);
      }
      if (m.a === myId && m.v !== myId) { showHitmarker(m.hs ? '#ffd23f' : '#ffffff'); snd.hit(m.hs); }
      break;
    }
    case 'kill': {
      applyScores(m.s);
      addKillfeed(m);
      if (m.v === myId) die(m.k);
      else {
        const r = remotes.get(m.v);
        if (r) {
          r.alive = false;
          featherBurst(r.pos.clone().setY(r.pos.y + 0.6), info.get(m.v)?.color || '#fff', 40);
          snd.bawk(r.pos.toArray(), true);
        }
        if (m.k === myId) {
          showHitmarker('#ff3344');
          snd.kill();
          showNotice(`${m.hs ? '🎯 HEADSHOT! ' : ''}You fried ${info.get(m.v)?.name || '???'}`);
        }
      }
      break;
    }
    case 'boom':
      explosion(m.p);
      break;
    case 'pick':
      pickupMeshes[m.i].visible = !!m.a;
      if (m.by === myId) { me.hp = m.hp; me.eggs = m.eggs; snd.pickup(); }
      break;
    case 'chat':
      addChat(`${nameHTML(m.id)}: ${esc(m.m)}`);
      break;
    case 'matchEnd':
      matchOver = true;
      applyScores(m.s);
      setHTML($('matchEnd'), `<h1>🏆 ${esc(m.name)} WINS!</h1><p style="margin:6px 0 14px;color:#ccc">${m.winner === myId ? 'Winner winner, chicken dinner!' : 'Next round starts in 10 seconds…'}</p>${scoreTable()}`);
      $('matchEnd').classList.remove('hidden');
      $('death').classList.add('hidden');
      break;
    case 'matchStart':
      matchOver = false;
      applyScores(m.s);
      $('matchEnd').classList.add('hidden');
      addChat('🐔 <i>New round! Go go go!</i>');
      break;
  }
}

// =====================================================================
// HUD
// =====================================================================
let hitmarkerT = 0, noticeTimer = 0;
function showHitmarker(color) {
  const h = $('hitmarker');
  h.style.color = color;
  hitmarkerT = 0.25;
}
function showNotice(text) {
  const n = $('notice');
  n.textContent = text;
  n.style.opacity = 1;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (n.style.opacity = 0), 1800);
}
function addKillfeed(m) {
  const feed = $('killfeed');
  const d = document.createElement('div');
  const weapon = `<span class="wpn">[${esc(WEAPON_LABELS[m.w] || '?')}${m.hs ? ' 🎯' : ''}]</span>`;
  d.innerHTML = m.k && m.k !== m.v ? `${nameHTML(m.k)}${weapon}${nameHTML(m.v)}` : `${nameHTML(m.v)}${weapon}<i>cooked themselves</i>`;
  if (m.k === myId || m.v === myId) d.classList.add('mine');
  feed.appendChild(d);
  while (feed.children.length > 6) feed.firstChild.remove();
  setTimeout(() => d.remove(), 6000);
}
function scoreTable() {
  const rows = [...info.entries()].sort((a, b) => b[1].kills - a[1].kills || a[1].deaths - b[1].deaths);
  return '<table><tr><th>Chicken</th><th>Kills</th><th>Deaths</th></tr>' + rows.map(([id, p]) =>
    `<tr class="${id === myId ? 'me' : ''}"><td><span class="dotc" style="background:${p.color}"></span>${esc(p.name)}</td><td>${p.kills}</td><td>${p.deaths}</td></tr>`).join('') + '</table>';
}

const slots = $('slots');
slots.innerHTML = WEAPONS.map((w, i) => `<div>${i + 1} ${w.name}</div>`).join('');

function updateHUD(dt, now) {
  if (!joined) return;
  const W = WEAPONS[me.w];
  const hp = Math.max(0, Math.round(me.hp));
  setText($('hpNum'), String(hp));
  const fill = $('hpFill');
  fill.style.width = `${hp}%`;
  fill.style.background = hp < 30 ? 'linear-gradient(90deg,#e63946,#ff6b6b)' : 'linear-gradient(90deg,#57cc99,#80ed99)';
  setText($('eggs'), `${'🥚'.repeat(me.eggs)}${me.eggs ? '' : 'no eggs'}  [G]`);
  setHTML($('ammo'), me.reloadUntil ? '… <small>/ ' + W.mag + '</small>' : `${me.ammo[me.w]} <small>/ ${W.mag}</small>`);
  setText($('weaponName'), W.name);
  [...slots.children].forEach((el, i) => el.classList.toggle('on', i === me.w));
  $('reloadMsg').classList.toggle('hidden', !me.reloadUntil || !me.alive);

  const scoped = me.alive && me.aiming && me.w === 3 && !thirdPerson;
  $('scope').classList.toggle('hidden', !scoped);
  const ch = $('crosshair');
  ch.style.display = scoped || !me.alive ? 'none' : '';
  ch.style.setProperty('--g', `${Math.round(3 + currentSpread() * 260)}px`);

  hitmarkerT = Math.max(0, hitmarkerT - dt);
  $('hitmarker').style.opacity = hitmarkerT > 0 ? 1 : 0;

  damageFlash = Math.max(0, damageFlash - dt * 1.5);
  $('vignette').style.opacity = Math.max(damageFlash, me.alive && me.hp < 30 ? 0.35 + Math.sin(now * 5) * 0.1 : 0);

  let leader = null;
  for (const p of info.values()) if (!leader || p.kills > leader.kills) leader = p;
  setText($('topInfo'), `First to ${scoreLimit} kills  ·  Leader: ${leader ? `${leader.name} (${leader.kills})` : '-'}  ·  ${info.size} chicken${info.size === 1 ? '' : 's'}`);

  const showBoard = keys.Tab && !matchOver;
  $('scoreboard').classList.toggle('hidden', !showBoard);
  if (showBoard) setHTML($('scoreboard'), `<h2>🐔 Scoreboard</h2>${scoreTable()}`);

  if (!me.alive && !matchOver) {
    const left = Math.max(0, RESPAWN_MS / 1000 - (now - me.deathTime));
    setText($('deathTimer'), `Respawning in ${left.toFixed(1)}s`);
  }
}

// =====================================================================
// Per-frame updates
// =====================================================================
function lerpAngle(a, b, k) {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * k;
}

function updateRemotes(dt, now) {
  const k = 1 - Math.exp(-dt * 15);
  for (const r of remotes.values()) {
    const px = r.pos.x, pz = r.pos.z;
    if (r.pos.distanceTo(r.target) > 6) r.pos.copy(r.target); else r.pos.lerp(r.target, k);
    r.ry = lerpAngle(r.ry, r.tRy, k);
    r.rx += (r.tRx - r.rx) * k;
    r.vy += ((r.pos.y - r.prevY) / Math.max(dt, 1e-3) - r.vy) * 0.3;
    r.prevY = r.pos.y;
    const hs = Math.hypot(r.pos.x - px, r.pos.z - pz) / Math.max(dt, 1e-3);
    const g = r.ch.group;
    g.visible = r.alive;
    g.position.copy(r.pos);
    g.rotation.y = r.ry;
    r.ch.head.rotation.x = r.rx * 0.5;
    r.ch.gunHolder.rotation.x = r.rx;
    animateChicken(r.ch, hs, r.vy, dt, now);
  }
  const ke = 1 - Math.exp(-dt * 20);
  for (const em of eggMeshes.values()) {
    em.position.lerp(em.userData.target, ke);
    em.rotation.x += dt * 8; em.rotation.z += dt * 5;
  }
  for (const pm of pickupMeshes) {
    pm.rotation.y += dt * 1.5;
    pm.children[0].position.y = Math.sin(now * 2) * 0.1;
  }
  if (myChicken) {
    myChicken.group.visible = thirdPerson && me.alive && joined;
    myChicken.group.position.copy(me.pos);
    myChicken.group.rotation.y = me.yaw;
    myChicken.head.rotation.x = me.pitch * 0.5;
    myChicken.gunHolder.rotation.x = me.pitch;
    myChicken.setWeapon(me.w);
  }
}

function updateCamera(dt, now) {
  if (!joined) {
    const a = now * 0.05;
    camera.position.set(Math.sin(a) * 48, 24, Math.cos(a) * 48);
    camera.lookAt(0, 2, 0);
  } else if (!me.alive) {
    const a = now * 0.4;
    camera.position.set(deathPos.x + Math.sin(a) * 5, deathPos.y + 3.5, deathPos.z + Math.cos(a) * 5);
    camera.lookAt(deathPos.x, deathPos.y + 0.5, deathPos.z);
  } else {
    if (me.onGround && me.pos.y > me.viewY && me.pos.y - me.viewY < 0.7) me.viewY += (me.pos.y - me.viewY) * Math.min(1, dt * 15);
    else me.viewY = me.pos.y;
    const sh = shake * 0.03;
    camera.rotation.set(me.pitch + rnd(-sh, sh), me.yaw + rnd(-sh, sh), 0);
    const bob = thirdPerson ? 0 : Math.sin(me.bob * 2) * 0.035 * Math.min(1, Math.hypot(me.vel.x, me.vel.z) / 6);
    const eye = new THREE.Vector3(me.pos.x, me.viewY + EYE + bob, me.pos.z);
    if (thirdPerson) {
      const fwd = aimForward();
      const right = new THREE.Vector3(Math.cos(me.yaw), 0, -Math.sin(me.yaw));
      const offset = fwd.clone().multiplyScalar(-3.2).addScaledVector(right, 0.7).add(new THREE.Vector3(0, 0.35, 0));
      const len = offset.length();
      offset.divideScalar(len);
      const t = raycastWorld(eye.toArray(), offset.toArray(), len);
      camera.position.copy(eye).addScaledVector(offset, Math.max(0.2, t - 0.25));
    } else {
      camera.position.copy(eye);
    }
    const W = WEAPONS[me.w];
    const sprinting = keys.ShiftLeft && Math.hypot(me.vel.x, me.vel.z) > 7;
    const targetFov = me.aiming ? W.zoom : sprinting ? BASE_FOV + 6 : BASE_FOV;
    if (Math.abs(camera.fov - targetFov) > 0.05) {
      camera.fov += (targetFov - camera.fov) * Math.min(1, dt * 16);
      camera.updateProjectionMatrix();
    }
  }
  if (!me.alive && camera.fov !== BASE_FOV) { camera.fov = BASE_FOV; camera.updateProjectionMatrix(); }
  camera.getWorldDirection(camFwd);
}

function updateViewmodel(dt, now) {
  const W = WEAPONS[me.w];
  const scoped = me.aiming && me.w === 3;
  vm.visible = joined && me.alive && !thirdPerson && !scoped;
  if (!vm.visible) return;
  vmGuns.forEach((g, i) => (g.visible = i === me.w));
  vmState.kick = Math.max(0, vmState.kick - dt * 8);
  vmState.swayX *= Math.max(0, 1 - dt * 8);
  vmState.swayY *= Math.max(0, 1 - dt * 8);
  const ads = me.aiming;
  let x = ads ? 0 : 0.3, y = ads ? -0.17 : -0.28, z = ads ? -0.45 : -0.6;
  let rotX = 0;
  if (me.reloadUntil) {
    const p = 1 - (me.reloadUntil - now) / W.reload;
    const s = Math.sin(Math.max(0, Math.min(1, p)) * Math.PI);
    y -= 0.22 * s; rotX -= 0.7 * s;
  }
  if (now < me.switchUntil) y -= 0.35 * ((me.switchUntil - now) / 0.35);
  const speed = Math.hypot(me.vel.x, me.vel.z);
  const bobAmp = me.onGround ? Math.min(1, speed / 6) : 0;
  x += Math.sin(me.bob) * 0.012 * bobAmp - vmState.swayX * 0.03;
  y += Math.abs(Math.cos(me.bob)) * 0.012 * bobAmp + vmState.swayY * 0.03;
  z += vmState.kick * 0.08;
  rotX += vmState.kick * 0.12;
  vm.position.lerp(new THREE.Vector3(x, y, z), Math.min(1, dt * 20));
  vm.rotation.set(rotX, -vmState.swayX * 0.1, 0);
  vmFlash.position.set(0, 0.03, W.muzzle - 0.08);
  vmFlash.visible = now < vmState.flashUntil;
  vmFlash.material.rotation = Math.random() * 6;
}

function sendState(now) {
  if (!joined || !me.alive || now - lastSend < 0.05) return;
  lastSend = now;
  send({ t: 's', p: [+me.pos.x.toFixed(3), +me.pos.y.toFixed(3), +me.pos.z.toFixed(3)], ry: +me.yaw.toFixed(3), rx: +me.pitch.toFixed(3), w: me.w });
}

// =====================================================================
// Menu
// =====================================================================
function showMenu() { $('menu').classList.remove('hidden'); }
function initMenu() {
  $('name').value = settings.name;
  $('limitTxt').textContent = SCORE_LIMIT;
  const colors = $('colors');
  colors.innerHTML = COLORS.map((c) => `<div class="swatch${c === settings.color ? ' sel' : ''}" data-c="${c}" style="background:${c}"></div>`).join('');
  colors.addEventListener('click', (e) => {
    const c = e.target.dataset?.c;
    if (!c) return;
    settings.color = c;
    colors.querySelectorAll('.swatch').forEach((s) => s.classList.toggle('sel', s.dataset.c === c));
  });
  $('sens').value = settings.sens;
  $('vol').value = settings.vol;
  $('sens').addEventListener('input', (e) => { settings.sens = +e.target.value; store.set('cg_sens', settings.sens); });
  $('vol').addEventListener('input', (e) => {
    settings.vol = +e.target.value; store.set('cg_vol', settings.vol);
    if (master) master.gain.value = settings.vol;
  });
  $('name').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('play').click(); });
  $('play').addEventListener('click', () => {
    initAudio();
    if (!joined) {
      settings.name = $('name').value.trim().slice(0, 16);
      store.set('cg_name', settings.name);
      store.set('cg_color', settings.color);
      if (!ws) connect();
    }
    lock();
  });
}
initMenu();

// =====================================================================
// Main loop
// =====================================================================
let last = performance.now();
function frame(t) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (t - last) / 1000);
  last = t;
  const now = t / 1000;

  if (joined && me.alive) {
    updateLocal(dt, now);
    if (me.autoReloadAt && now >= me.autoReloadAt) { me.autoReloadAt = 0; startReload(now); }
  }
  updateRemotes(dt, now);
  updateCamera(dt, now);
  updateViewmodel(dt, now);
  updateParticles(dt);
  updateEffects(dt);
  updateHUD(dt, now);
  sendState(now);

  renderer.clear();
  renderer.render(scene, camera);
  if (vm.visible) {
    renderer.clearDepth();
    renderer.render(vmScene, vmCamera);
  }
}
requestAnimationFrame(frame);
