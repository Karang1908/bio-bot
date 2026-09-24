import * as THREE from "three";
import type { HFieldDef, WaterDef } from "../lib/types";

/** Heights (metres) of one heightfield, read from the scene blob. Row index follows +y. */
export function hfieldHeights(buf: ArrayBuffer, h: HFieldDef): Float32Array {
  return new Float32Array(buf, h.offset, h.nrow * h.ncol);
}

const SAND = new THREE.Color("#c9b68c");
/** The plain grass colour: the terrain fades to it at the map edge, and the ground beyond uses it. */
export const GRASS_HEX = "#62843f";
const GRASS = new THREE.Color(GRASS_HEX);
const DEEP_GRASS = new THREE.Color("#46652d");
const ROCK = new THREE.Color("#7c7669");
const MUD = new THREE.Color("#6b5a41");
const DRY_GRASS = new THREE.Color("#8a9450");

/** Smooth 2D value noise in [0, 1] with features about `scale` metres across. */
function valueNoise(x: number, y: number, scale: number): number {
  const fx = x / scale, fy = y / scale;
  const ix = Math.floor(fx), iy = Math.floor(fy);
  const tx = fx - ix, ty = fy - iy;
  const hash = (a: number, b: number) => {
    const v = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
    return v - Math.floor(v);
  };
  const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

/** Terrain mesh in MuJoCo coordinates (z up), coloured by height, slope and soft patches. */
export function terrainGeometry(buf: ArrayBuffer, h: HFieldDef, water: WaterDef[]): THREE.BufferGeometry {
  const heights = hfieldHeights(buf, h);
  const geo = new THREE.PlaneGeometry(2 * h.halfX, 2 * h.halfY, h.ncol - 1, h.nrow - 1);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const dx = (2 * h.halfX) / (h.ncol - 1);
  const c = new THREE.Color();
  for (let iy = 0; iy < h.nrow; iy++) {
    const row = h.nrow - 1 - iy; // PlaneGeometry starts at +y; MuJoCo row 0 is -y
    for (let ix = 0; ix < h.ncol; ix++) {
      const k = iy * h.ncol + ix;
      const z = heights[row * h.ncol + ix];
      pos.setZ(k, z);
      const zl = heights[row * h.ncol + Math.max(ix - 1, 0)];
      const zr = heights[row * h.ncol + Math.min(ix + 1, h.ncol - 1)];
      const zd = heights[Math.max(row - 1, 0) * h.ncol + ix];
      const zu = heights[Math.min(row + 1, h.nrow - 1) * h.ncol + ix];
      const slope = Math.hypot(zr - zl, zu - zd) / (2 * dx);
      const x = pos.getX(k), y = pos.getY(k);
      const nearWater = water.some((w) => ((x - w.cx) / (w.rx + 4)) ** 2 + ((y - w.cy) / (w.ry + 4)) ** 2 < 1);
      // Soft patches of lusher and drier grass (no regular pattern), fading to plain grass at the edge.
      const patch = 0.65 * valueNoise(x, y, 13) + 0.35 * valueNoise(x + 57, y - 31, 4.5);
      const edge = THREE.MathUtils.clamp((h.halfX - Math.max(Math.abs(x), Math.abs(y))) / 8, 0, 1);
      if (nearWater && z < 0.15) c.copy(z < -0.5 ? MUD : SAND);
      else {
        c.copy(GRASS).lerp(DEEP_GRASS, THREE.MathUtils.clamp(z / 4, 0, 1));
        c.lerp(patch > 0.5 ? DEEP_GRASS : DRY_GRASS, Math.abs(patch - 0.5) * 0.9 * edge);
      }
      c.lerp(ROCK, THREE.MathUtils.clamp((slope - 0.35) * 2.2, 0, 0.85));
      colors[k * 3] = c.r;
      colors[k * 3 + 1] = c.g;
      colors[k * 3 + 2] = c.b;
    }
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}
