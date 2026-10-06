// Shared between server and client.

export const WEAPONS = [
  { name: 'Pistol',     dmg: 26, rate: 0.25, mag: 12, reload: 1.1, spread: 0.012, pellets: 1, auto: false, range: 150, recoil: 0.014, falloff: false, zoom: 55, muzzle: -0.3 },
  { name: 'Shotgun',    dmg: 14, rate: 0.85, mag: 6,  reload: 1.9, spread: 0.075, pellets: 9, auto: false, range: 45,  recoil: 0.055, falloff: true,  zoom: 62, muzzle: -0.65 },
  { name: 'Cluck-47',   dmg: 17, rate: 0.10, mag: 30, reload: 1.7, spread: 0.022, pellets: 1, auto: true,  range: 150, recoil: 0.009, falloff: false, zoom: 50, muzzle: -0.62 },
  { name: 'Eggsniper',  dmg: 90, rate: 1.30, mag: 5,  reload: 2.2, spread: 0.08,  pellets: 1, auto: false, range: 300, recoil: 0.06,  falloff: false, zoom: 18, muzzle: -0.95, scopedSpread: 0 },
];

export const EGG_WEAPON = 4; // weapon index used for egg-grenade kills
export const WEAPON_LABELS = ['Pistol', 'Shotgun', 'Cluck-47', 'Eggsniper', 'Egg Grenade'];

export const EGG = { radius: 5, dmg: 110, fuse: 2.2, max: 3, speed: 17 };
export const PLAYER = { R: 0.35, H: 1.3, EYE: 1.1, STEP: 0.55, HP: 100 };
export const HEADSHOT = 2;
export const SCORE_LIMIT = 20;
export const RESPAWN_MS = 3000;
