import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';
import { BOXES, SPAWNS, PICKUPS, ARENA, raycastWorld, chickenHit } from './public/shared/map.js';
import { WEAPONS, EGG, EGG_WEAPON, PLAYER, HEADSHOT, SCORE_LIMIT, RESPAWN_MS } from './public/shared/weapons.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(__dirname, 'public');
const VENDOR = path.join(__dirname, 'node_modules', 'three', 'build');
const PORT = Number(process.env.PORT) || 3000;
const TICK = 1 / 30;
const COLORS = ['#ffffff', '#ffd23f', '#8b5a2b', '#333333', '#e63946', '#4ea8de', '#57cc99', '#ff8fab', '#9b5de5', '#ff9f1c'];

// ---------------- Static file server ----------------
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.ico': 'image/x-icon', '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  let url;
  try { url = decodeURIComponent(req.url.split('?')[0]); } catch { res.writeHead(400); return res.end(); }
  let root = PUBLIC, rel = url;
  if (url.startsWith('/vendor/')) { root = VENDOR; rel = url.slice('/vendor'.length); }
  if (rel === '/') rel = '/index.html';
  const file = path.join(root, path.normalize(rel));
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  });
});

// ---------------- Game state ----------------
const players = new Map();
let nextId = 1;
let eggs = [];
let nextEggId = 1;
let matchOver = false;
const pickups = PICKUPS.map((pos, i) => ({ i, pos, active: true, respawnAt: 0 }));

const r2 = (x) => Math.round(x * 100) / 100;
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const isVec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
const normalize = (v) => { const l = Math.hypot(v[0], v[1], v[2]); return l > 1e-6 ? [v[0] / l, v[1] / l, v[2] / l] : null; };
const eyeOf = (p) => [p.pos[0], p.pos[1] + PLAYER.EYE, p.pos[2]];

function send(ws, msg) { if (ws.readyState === 1) ws.send(JSON.stringify(msg)); }
function broadcast(msg, except) {
  const s = JSON.stringify(msg);
  for (const p of players.values()) if (p !== except && p.ws.readyState === 1) p.ws.send(s);
}
const publicInfo = (p) => ({ id: p.id, name: p.name, color: p.color, kills: p.kills, deaths: p.deaths, alive: p.alive, pos: p.pos, ry: p.ry, w: p.w });
const scores = () => [...players.values()].map((p) => [p.id, p.kills, p.deaths]);

function pickSpawn() {
  let best = SPAWNS[0], bestScore = -1;
  for (const s of SPAWNS) {
    let nearest = 200;
    for (const p of players.values()) if (p.alive) nearest = Math.min(nearest, Math.hypot(p.pos[0] - s[0], p.pos[2] - s[2]));
    const score = nearest + Math.random() * 10;
    if (score > bestScore) { bestScore = score; best = s; }
  }
  return best;
}

function spawn(p) {
  const s = pickSpawn();
  p.pos = [s[0], s[1], s[2]];
  p.ry = Math.atan2(s[0], s[2]); // face the arena center
  p.rx = 0;
  p.hp = PLAYER.HP;
  p.alive = true;
  p.eggs = EGG.max;
  p.protectUntil = Date.now() + 1500;
  broadcast({ t: 'spawn', id: p.id, p: p.pos, ry: r2(p.ry), hp: p.hp, eggs: p.eggs });
}

function damage(victim, attacker, amount, headshot, weapon, point) {
  if (!victim.alive || matchOver || amount <= 0) return;
  if (Date.now() < victim.protectUntil) return;
  victim.hp -= amount;
  broadcast({ t: 'hit', v: victim.id, a: attacker ? attacker.id : 0, hp: Math.max(0, victim.hp), hs: headshot, d: amount, p: point.map(r2) });
  if (victim.hp <= 0) kill(victim, attacker, weapon, headshot);
}

function kill(victim, killer, weapon, headshot) {
  victim.alive = false;
  victim.hp = 0;
  victim.deaths++;
  if (killer && killer !== victim) killer.kills++;
  broadcast({ t: 'kill', k: killer ? killer.id : 0, v: victim.id, w: weapon, hs: headshot, s: scores() });
  console.log(`${killer ? killer.name : '???'} -> ${victim.name} (${weapon})`);
  if (killer && killer !== victim && killer.kills >= SCORE_LIMIT) return endMatch(killer);
  clearTimeout(victim.respawnTimer);
  victim.respawnTimer = setTimeout(() => {
    if (players.has(victim.id) && !matchOver && !victim.alive) spawn(victim);
  }, RESPAWN_MS);
}

function endMatch(winner) {
  matchOver = true;
  eggs = [];
  broadcast({ t: 'matchEnd', winner: winner.id, name: winner.name, s: scores() });
  console.log(`Match over! ${winner.name} wins.`);
  setTimeout(() => {
    matchOver = false;
    for (const p of players.values()) { p.kills = 0; p.deaths = 0; clearTimeout(p.respawnTimer); }
    broadcast({ t: 'matchStart', s: scores() });
    for (const p of players.values()) spawn(p);
  }, 10000);
}

function explode(e) {
  e.dead = true;
  broadcast({ t: 'boom', p: e.pos.map(r2) });
  const owner = players.get(e.owner) || null;
  for (const q of players.values()) {
    if (!q.alive) continue;
    const c = [q.pos[0], q.pos[1] + 0.6, q.pos[2]];
    const d = dist(e.pos, c);
    if (d >= EGG.radius) continue;
    if (d > 0.3) {
      const dir = normalize([c[0] - e.pos[0], c[1] - e.pos[1], c[2] - e.pos[2]]);
      if (raycastWorld(e.pos, dir, d) < d - 0.4) continue; // wall in between
    }
    let amount = EGG.dmg * (1 - d / EGG.radius);
    if (q === owner) amount *= 0.5;
    damage(q, owner, Math.round(amount), false, EGG_WEAPON, c);
  }
}

function eggInside(p) {
  for (const b of BOXES) {
    if (p[0] > b.min[0] - 0.1 && p[0] < b.max[0] + 0.1 && p[1] > b.min[1] - 0.1 && p[1] < b.max[1] + 0.1 &&
        p[2] > b.min[2] - 0.1 && p[2] < b.max[2] + 0.1) return true;
  }
  return false;
}

// ---------------- Messages ----------------
function createPlayer(ws, m) {
  let name = String(m.name || '').replace(/[<>]/g, '').trim().slice(0, 16) || `Chicken${nextId}`;
  const color = /^#[0-9a-f]{6}$/i.test(m.color) ? m.color : COLORS[nextId % COLORS.length];
  const p = {
    id: nextId++, ws, name, color, pos: [0, 0, 0], ry: 0, rx: 0, w: 0, hp: 0, alive: false,
    kills: 0, deaths: 0, eggs: EGG.max, lastShot: 0, lastEgg: 0, protectUntil: 0, respawnTimer: null,
  };
  players.set(p.id, p);
  send(ws, {
    t: 'welcome', id: p.id, scoreLimit: SCORE_LIMIT, matchOver,
    players: [...players.values()].map(publicInfo),
    pickups: pickups.map((pk) => (pk.active ? 1 : 0)),
  });
  broadcast({ t: 'join', p: publicInfo(p) }, p);
  console.log(`${p.name} joined (${players.size} online)`);
  if (!matchOver) spawn(p);
  return p;
}

function handle(p, m) {
  const now = Date.now();
  switch (m.t) {
    case 's': {
      if (!p.alive || !isVec3(m.p)) return;
      p.pos = [Math.max(-ARENA, Math.min(ARENA, m.p[0])), Math.max(0, Math.min(30, m.p[1])), Math.max(-ARENA, Math.min(ARENA, m.p[2]))];
      if (Number.isFinite(m.ry)) p.ry = m.ry;
      if (Number.isFinite(m.rx)) p.rx = m.rx;
      if (Number.isInteger(m.w) && WEAPONS[m.w]) p.w = m.w;
      break;
    }
    case 'shoot': {
      const W = WEAPONS[m.w];
      if (!W || !p.alive || matchOver || !isVec3(m.o) || !Array.isArray(m.d)) return;
      if (now - p.lastShot < W.rate * 1000 * 0.8) return;
      p.lastShot = now;
      const o = m.o;
      if (dist(o, eyeOf(p)) > 2.5) return;
      const ends = [];
      const hits = new Map();
      for (const raw of m.d.slice(0, W.pellets)) {
        if (!isVec3(raw)) continue;
        const d = normalize(raw);
        if (!d) continue;
        let t = raycastWorld(o, d, W.range), target = null, hs = false;
        for (const q of players.values()) {
          if (q === p || !q.alive) continue;
          const h = chickenHit(o, d, q.pos, q.ry);
          if (h.t < t) { t = h.t; target = q; hs = h.head; }
        }
        const end = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
        ends.push(end.map(r2));
        if (target) {
          let amount = W.dmg * (hs ? HEADSHOT : 1);
          if (W.falloff) amount *= Math.max(0.35, 1 - t / W.range);
          const e = hits.get(target) || { a: 0, hs: false, pt: end };
          e.a += amount; e.hs = e.hs || hs;
          hits.set(target, e);
        }
      }
      broadcast({ t: 'shot', id: p.id, w: m.w, e: ends }, p);
      for (const [q, e] of hits) damage(q, p, Math.round(e.a), e.hs, m.w, e.pt);
      break;
    }
    case 'egg': {
      if (!p.alive || matchOver || p.eggs <= 0 || !isVec3(m.o) || !isVec3(m.v)) return;
      if (now - p.lastEgg < 600 || dist(m.o, eyeOf(p)) > 2.5) return;
      p.lastEgg = now;
      p.eggs--;
      let v = m.v;
      const sp = Math.hypot(v[0], v[1], v[2]), max = EGG.speed * 1.6;
      if (sp > max) v = v.map((x) => (x * max) / sp);
      eggs.push({ id: nextEggId++, owner: p.id, pos: [...m.o], vel: [...v], fuse: EGG.fuse, age: 0 });
      break;
    }
    case 'chat': {
      const text = String(m.m || '').slice(0, 140).trim();
      if (text) broadcast({ t: 'chat', id: p.id, m: text });
      break;
    }
  }
}

// ---------------- Networking ----------------
const wss = new WebSocketServer({ server });
wss.on('connection', (ws) => {
  let me = null;
  ws.on('message', (raw) => {
    let m;
    try { m = JSON.parse(raw); } catch { return; }
    if (!m || typeof m.t !== 'string') return;
    if (!me) { if (m.t === 'join') me = createPlayer(ws, m); return; }
    handle(me, m);
  });
  ws.on('close', () => {
    if (!me) return;
    clearTimeout(me.respawnTimer);
    players.delete(me.id);
    broadcast({ t: 'leave', id: me.id });
    console.log(`${me.name} left (${players.size} online)`);
  });
});

// ---------------- Simulation tick ----------------
let lastTick = Date.now();
setInterval(() => {
  const now = Date.now();
  // Timers on Windows are coarse (~16ms), so simulate with the real elapsed time.
  const elapsed = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  const sub = Math.max(1, Math.ceil(elapsed / (1 / 60)));
  const dt = elapsed / sub;

  for (const e of eggs) {
    e.age += elapsed;
    e.fuse -= elapsed;
    for (let s = 0; s < sub; s++) {
      e.vel[1] -= 20 * dt;
      for (let ax = 0; ax < 3; ax++) {
        const old = e.pos[ax];
        e.pos[ax] += e.vel[ax] * dt;
        if (eggInside(e.pos)) {
          e.pos[ax] = old;
          e.vel[ax] *= -0.45;
          e.vel[(ax + 1) % 3] *= 0.8;
          e.vel[(ax + 2) % 3] *= 0.8;
        }
      }
      if (e.pos[1] < 0.12) { e.pos[1] = 0.12; e.vel[1] *= -0.45; e.vel[0] *= 0.75; e.vel[2] *= 0.75; }
    }
    for (const q of players.values()) {
      if (!q.alive || (q.id === e.owner && e.age < 0.4)) continue;
      if (dist(e.pos, [q.pos[0], q.pos[1] + 0.6, q.pos[2]]) < 0.65) { e.fuse = 0; break; }
    }
    if (e.fuse <= 0) explode(e);
  }
  eggs = eggs.filter((e) => !e.dead);

  for (const pk of pickups) {
    if (!pk.active) {
      if (now >= pk.respawnAt) { pk.active = true; broadcast({ t: 'pick', i: pk.i, a: 1 }); }
      continue;
    }
    for (const p of players.values()) {
      if (!p.alive || (p.hp >= PLAYER.HP && p.eggs >= EGG.max)) continue;
      if (Math.hypot(p.pos[0] - pk.pos[0], p.pos[2] - pk.pos[2]) < 1.2 && Math.abs(p.pos[1] - pk.pos[1]) < 1.5) {
        p.hp = Math.min(PLAYER.HP, p.hp + 50);
        p.eggs = Math.min(EGG.max, p.eggs + 1);
        pk.active = false;
        pk.respawnAt = now + 15000;
        broadcast({ t: 'pick', i: pk.i, a: 0, by: p.id, hp: p.hp, eggs: p.eggs });
        break;
      }
    }
  }

  if (players.size) {
    broadcast({
      t: 'snap',
      p: [...players.values()].map((p) => [p.id, r2(p.pos[0]), r2(p.pos[1]), r2(p.pos[2]), r2(p.ry), r2(p.rx), p.w, p.alive ? 1 : 0]),
      e: eggs.map((e) => [e.id, r2(e.pos[0]), r2(e.pos[1]), r2(e.pos[2])]),
    });
  }
}, TICK * 1000);

server.listen(PORT, '0.0.0.0', () => {
  console.log('\n  🐔  CHICKEN GUN server is running!\n');
  console.log(`  On this computer:   http://localhost:${PORT}`);
  for (const list of Object.values(os.networkInterfaces())) {
    for (const a of list || []) {
      if (a.family === 'IPv4' && !a.internal) console.log(`  Friends on LAN:     http://${a.address}:${PORT}`);
    }
  }
  console.log('\n  (If friends cannot connect, allow Node.js through Windows Firewall for private networks.)\n');
});
