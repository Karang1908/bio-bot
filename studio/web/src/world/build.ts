import * as THREE from "three";
import { mergeGeometries, toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import type { GeomDef, SceneDef } from "../lib/types";
import {
  type Facade, facadeTexture, floorTexture, pavingTexture, plasterTexture, roadTexture, roofTexture, rugTexture,
} from "./textures";

const CREASE = THREE.MathUtils.degToRad(35);

export interface BuiltWorld {
  bodies: THREE.Group[]; // index = MuJoCo body id
  collision: THREE.Object3D[];
  agentGroups: THREE.Group[]; // groups that belong to a body in the lineup (for picking)
  dispose: () => void;
}

type Finish = () => Partial<THREE.MeshPhysicalMaterialParameters> & { physical?: boolean };

/** Procedural surfaces for props that carry no texture in MuJoCo, matched by geom name. */
function finishFor(g: GeomDef): { key: string; make: Finish } | null {
  if (g.name === "room_floor") return { key: g.name, make: () => ({ map: floorTexture(), roughness: 0.62 }) };
  if (g.name === "floor_rug") return { key: g.name, make: () => ({ map: rugTexture(), roughness: 0.95 }) };
  if (g.name.startsWith("road_")) {
    const along = g.size[1] > g.size[0] ? "v" : "u";
    const repeat = Math.round(Math.max(g.size[0], g.size[1]) / 2.5);
    return { key: `road-${along}-${repeat}`, make: () => ({ map: roadTexture(along, repeat), roughness: 0.9 }) };
  }
  if (g.name.startsWith("water")) {
    return {
      key: "water",
      make: () => ({ physical: true, color: new THREE.Color("#2b6f86"), roughness: 0.06, metalness: 0.0,
        transparent: true, opacity: 0.72, depthWrite: false, clearcoat: 1, clearcoatRoughness: 0.05 }),
    };
  }
  if (g.name.includes("leaves")) {
    return { key: `leaves-${g.id % 3}`, make: () => ({
      color: new THREE.Color(["#5c8f3f", "#6a9a45", "#527f38"][g.id % 3]), roughness: 0.9, vertexColors: true }) };
  }
  if (g.name.startsWith("sidewalk_")) return { key: "paving", make: () => ({ map: pavingTexture(), roughness: 0.92 }) };
  if (g.name.startsWith("wall_")) return { key: "plaster", make: () => ({ map: plasterTexture(), roughness: 0.9 }) };
  if (/^lamp\d+_head$/.test(g.name)) {
    return { key: "lamp-head", make: () => ({ color: new THREE.Color("#fff4dc"), emissive: new THREE.Color("#ffe2a8"),
      emissiveIntensity: 0.9, roughness: 0.4 }) };
  }
  return null;
}

/** "bldg_glass_1" -> "glass"; null for anything that isn't a building shell. */
function facadeOf(name: string): Facade | null {
  const m = /^bldg_(brick|concrete|glass)_\d+$/.exec(name);
  return m ? (m[1] as Facade) : null;
}

/** Building shell: facade on the four sides, gravel on the roof (BoxGeometry face order +x -x +y -y +z -z). */
function buildingMaterials(kind: Facade, cache: Map<string, THREE.Material>): THREE.Material[] {
  const key = `facade:${kind}`;
  let side = cache.get(key);
  if (!side) {
    side = new THREE.MeshStandardMaterial({
      map: facadeTexture(kind),
      roughness: kind === "glass" ? 0.18 : kind === "concrete" ? 0.8 : 0.88,
      metalness: kind === "glass" ? 0.45 : 0,
    });
    cache.set(key, side);
  }
  let roof = cache.get("roof");
  if (!roof) {
    roof = new THREE.MeshStandardMaterial({ map: roofTexture(), roughness: 0.95 });
    cache.set("roof", roof);
  }
  return [side, side, side, side, roof, roof];
}

/** Metres-based box UVs (u, v = distance / tile), so a texture keeps its real size on faces of any shape.
 *  Vertical faces get v = height above the box's bottom, so building floors line up with the ground. */
function worldUVs(geo: THREE.BufferGeometry, half: [number, number, number], tile: number): void {
  const p = geo.attributes.position, n = geo.attributes.normal, uv = geo.attributes.uv;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i) + half[0], y = p.getY(i) + half[1], z = p.getZ(i) + half[2];
    const [u, v] = Math.abs(n.getX(i)) > 0.5 ? [y, z] : Math.abs(n.getY(i)) > 0.5 ? [x, z] : [x, y];
    uv.setXY(i, u / tile, v / tile);
  }
  uv.needsUpdate = true;
}

/** A leafy crown of a few overlapping lobes, unit size, shaded darker underneath (vertex colours). */
function crownGeometry(seed: number): THREE.BufferGeometry {
  let s = (seed * 2654435761) >>> 0;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const lobes: [number, number, number, number][] = [[0, 0, 0, 0.78]];
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 + r() * 0.8, rad = 0.38 + r() * 0.18;
    lobes.push([Math.cos(a) * rad, Math.sin(a) * rad, (r() - 0.35) * 0.55, 0.42 + r() * 0.16]);
  }
  lobes.push([(r() - 0.5) * 0.2, (r() - 0.5) * 0.2, 0.5, 0.45]);
  const parts = lobes.map(([x, y, z, rad], k) => {
    const geo = new THREE.IcosahedronGeometry(rad, 3);
    const pos = geo.attributes.position;
    const col = new Float32Array(pos.count * 3);
    const tint = 0.9 + r() * 0.2;
    for (let i = 0; i < pos.count; i++) {
      const px = pos.getX(i), py = pos.getY(i), pz = pos.getZ(i);
      const bump = 1 + 0.07 * Math.sin(px * 17 + k) * Math.sin(py * 15 + k * 2) * Math.sin(pz * 13);
      pos.setXYZ(i, px * bump + x, py * bump + y, pz * bump + z);
      const up = THREE.MathUtils.clamp((pz + z + 0.9) / 1.8, 0, 1);  // 0 underneath, 1 on top
      const shade = (0.55 + 0.5 * up) * tint;
      col[i * 3] = shade;
      col[i * 3 + 1] = shade;
      col[i * 3 + 2] = shade * 0.92;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return geo;
  });
  const merged = mergeGeometries(parts)!;
  parts.forEach((p) => p.dispose());
  return merged;
}

function material(g: GeomDef, cache: Map<string, THREE.Material>): THREE.Material {
  const finish = finishFor(g);
  const key = finish ? `finish:${finish.key}` : `${g.rgba.join(",")}|${g.metallic}|${g.roughness}|${g.emission}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const [r, gg, b, a] = g.rgba;
  const color = new THREE.Color().setRGB(r, gg, b, THREE.SRGBColorSpace);
  let params: THREE.MeshPhysicalMaterialParameters = {
    color,
    metalness: g.metallic,
    roughness: g.roughness,
    transparent: a < 0.99,
    opacity: a,
    depthWrite: a >= 0.99,
    side: a < 0.99 ? THREE.DoubleSide : THREE.FrontSide,
  };
  let physical = g.name === "apple_fruit";
  if (finish) {
    const { physical: p, ...rest } = finish.make();
    params = { ...params, color: 0xffffff, ...rest };
    physical = physical || !!p;
  }
  if (g.emission > 0) {
    params.emissive = color;
    params.emissiveIntensity = g.emission;
  }
  // The apple gets a waxy skin; water a glassy coat; everything else stays plain PBR.
  const m = g.name === "apple_fruit"
    ? new THREE.MeshPhysicalMaterial({ ...params, clearcoat: 0.8, clearcoatRoughness: 0.25, roughness: 0.35 })
    : physical ? new THREE.MeshPhysicalMaterial(params) : new THREE.MeshStandardMaterial(params);
  cache.set(key, m);
  return m;
}

function geometry(g: GeomDef, scene: SceneDef, buf: ArrayBuffer): THREE.BufferGeometry | null {
  const [s0, s1, s2] = g.size;
  switch (g.type) {
    case "box": {
      const geo = new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2);
      if (facadeOf(g.name)) worldUVs(geo, [s0, s1, s2], 3);
      else if (g.name.startsWith("sidewalk_")) worldUVs(geo, [s0, s1, s2], 2);
      else if (g.name.startsWith("wall_")) worldUVs(geo, [s0, s1, s2], 2);
      return geo;
    }
    case "sphere":
      return new THREE.SphereGeometry(s0, 32, 16);
    case "ellipsoid": {
      const leafy = g.name.includes("leaves");
      const geo = leafy ? crownGeometry(g.id) : new THREE.SphereGeometry(1, 48, 24);
      geo.scale(s0, s1, s2);
      return geo;
    }
    case "capsule": {
      const geo = new THREE.CapsuleGeometry(s0, 2 * s1, 8, 24);
      geo.rotateX(Math.PI / 2); // three builds along Y; MuJoCo capsules run along local Z
      return geo;
    }
    case "cylinder": {
      const geo = new THREE.CylinderGeometry(s0, s0, 2 * s1, 32);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    case "mesh": {
      const m = scene.meshes.find((x) => x.id === g.mesh);
      if (!m) return null;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, m.vertOffset, m.vertCount * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, m.faceOffset, m.faceCount * 3), 1));
      return toCreasedNormals(geo, CREASE);
    }
    default:
      return null; // planes and heightfields: the ground is drawn separately
  }
}

export function buildWorld(scene: SceneDef, buf: ArrayBuffer): BuiltWorld {
  const bodies = scene.bodies.map((b) => {
    const grp = new THREE.Group();
    grp.name = b.name || `body${b.id}`;
    grp.userData = { body: b.id, agent: b.agent };
    return grp;
  });
  const cache = new Map<string, THREE.Material>();
  const meshGeoCache = new Map<number, THREE.BufferGeometry>();
  const collisionMat = new THREE.MeshBasicMaterial({ color: 0xffb000, wireframe: true, transparent: true, opacity: 0.35 });
  const collision: THREE.Object3D[] = [];
  const geos: THREE.BufferGeometry[] = [];

  for (const g of scene.geoms) {
    let geo: THREE.BufferGeometry | null;
    if (g.type === "mesh" && g.mesh !== null && meshGeoCache.has(g.mesh)) {
      geo = meshGeoCache.get(g.mesh)!;
    } else {
      geo = geometry(g, scene, buf);
      if (geo) geos.push(geo);
      if (geo && g.type === "mesh" && g.mesh !== null) meshGeoCache.set(g.mesh, geo);
    }
    if (!geo) continue;
    const kind = facadeOf(g.name);
    const mat = !g.visual ? collisionMat : kind ? buildingMaterials(kind, cache) : material(g, cache);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.name = g.name;
    mesh.userData = { body: g.body, agent: g.agent };
    if (g.visual) {
      mesh.castShadow = g.rgba[3] > 0.5 && !/^(water|road|sidewalk)/.test(g.name);
      mesh.receiveShadow = true;
    } else {
      mesh.visible = false;
      collision.push(mesh);
    }
    bodies[g.body].add(mesh);
  }

  return {
    bodies,
    collision,
    agentGroups: bodies.filter((b) => b.userData.agent),
    dispose: () => {
      geos.forEach((x) => x.dispose());
      cache.forEach((m) => m.dispose());
      collisionMat.dispose();
    },
  };
}
