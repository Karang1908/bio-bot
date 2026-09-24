import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitImpl } from "three-stdlib";
import { buildWorld, type BuiltWorld } from "./build";
import { GRASS_HEX, terrainGeometry } from "./terrain";
import { homeView } from "./views";
import { grassTexture, studioTexture } from "./textures";
import { camera as cam, flyTo, getState, poses, setState, useStore } from "../lib/store";
import type { SceneDef } from "../lib/types";
import { look } from "../ui/theme";

const GOAL = new THREE.Color("#FFFFFF");
const RING_RADIUS: Record<string, number> = { humanoid: 0.55, dog: 0.55, drone: 0.6, fly: 0.35, car: 1.25 };
const tmpQ0 = new THREE.Quaternion();
const tmpQ1 = new THREE.Quaternion();
const UP = new THREE.Vector3(0, 1, 0);

/** Ground height (m) under a point of the open world's heightfield (bilinear); 0 off the map or in the sandbox. */
function groundAt(scene: SceneDef, buf: ArrayBuffer, x: number, y: number): number {
  const h = scene.hfields[0];
  if (!h) return 0;
  const fx = ((x + h.halfX) / (2 * h.halfX)) * (h.ncol - 1), fy = ((y + h.halfY) / (2 * h.halfY)) * (h.nrow - 1);
  if (fx < 0 || fy < 0 || fx > h.ncol - 1 || fy > h.nrow - 1) return 0;
  const z = new Float32Array(buf, h.offset, h.nrow * h.ncol); // row index follows +y, as in terrain.ts
  const c0 = Math.min(Math.floor(fx), h.ncol - 2), r0 = Math.min(Math.floor(fy), h.nrow - 2);
  const tx = fx - c0, ty = fy - r0;
  const at = (r: number, c: number) => z[r * h.ncol + c];
  return (at(r0, c0) * (1 - tx) + at(r0, c0 + 1) * tx) * (1 - ty) + (at(r0 + 1, c0) * (1 - tx) + at(r0 + 1, c0 + 1) * tx) * ty;
}

/** MuJoCo (x, y, z) -> three (x, z, -y). */
const toThree = (x: number, y: number, z: number, out = new THREE.Vector3()) => out.set(x, z, -y);

/** MuJoCo is z-up; three is y-up. Everything MuJoCo lives inside this rotated frame. */
function MujocoFrame({ children }: { children: React.ReactNode }) {
  return <group rotation={[-Math.PI / 2, 0, 0]}>{children}</group>;
}

// ---------------------------------------------------------------------------- bodies
const HIGHLIGHT = new THREE.MeshStandardMaterial({
  color: "#7dd3fc", emissive: "#38bdf8", emissiveIntensity: 0.55, roughness: 0.4, transparent: true, opacity: 0.92,
});

function Bodies({ built }: { built: BuiltWorld }) {
  const showCollision = useStore((s) => s.showCollision);
  const hoverPart = useStore((s) => s.hoverPart);
  useEffect(() => {
    built.collision.forEach((o) => (o.visible = showCollision));
  }, [built, showCollision]);

  // Tint the hovered part (one MuJoCo body) without touching shared materials.
  useEffect(() => {
    if (hoverPart === null) return;
    const grp = built.bodies[hoverPart];
    if (!grp) return;
    const swapped: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    grp.children.forEach((c) => {
      const m = c as THREE.Mesh;
      if (m.isMesh && m.visible) {
        swapped.push([m, m.material]);
        m.material = HIGHLIGHT;
      }
    });
    return () => swapped.forEach(([m, mat]) => (m.material = mat));
  }, [built, hoverPart]);

  useFrame(() => {
    const { prev, curr, prevAt, currAt } = poses;
    if (!curr) return;
    const span = Math.max(currAt - prevAt, 1);
    const a = prev ? THREE.MathUtils.clamp((performance.now() - currAt) / span, 0, 1) : 1;
    const n = Math.min(built.bodies.length, curr.length / 7);
    for (let i = 1; i < n; i++) {
      const o = i * 7;
      const g = built.bodies[i];
      if (prev && prev.length === curr.length) {
        g.position.set(
          prev[o] + (curr[o] - prev[o]) * a,
          prev[o + 1] + (curr[o + 1] - prev[o + 1]) * a,
          prev[o + 2] + (curr[o + 2] - prev[o + 2]) * a,
        );
        tmpQ0.set(prev[o + 4], prev[o + 5], prev[o + 6], prev[o + 3]);
        tmpQ1.set(curr[o + 4], curr[o + 5], curr[o + 6], curr[o + 3]);
        g.quaternion.slerpQuaternions(tmpQ0, tmpQ1, a);
      } else {
        g.position.set(curr[o], curr[o + 1], curr[o + 2]);
        g.quaternion.set(curr[o + 4], curr[o + 5], curr[o + 6], curr[o + 3]);
      }
    }
  });

  return (
    <>
      {built.bodies.map((g, i) => (
        <primitive key={i} object={g} />
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------- ground
/** Haze that the far ground fades into, close to the sky's colour at the horizon. */
const HAZE = "#c9d4de";
const PLAIN = 1500; // metres: the ground around the map runs out to the haze, so there is no edge to see

function Ground({ scene, buf }: { scene: SceneDef; buf: ArrayBuffer }) {
  const hf = scene.hfields[0];
  const terrain = useMemo(() => (hf ? terrainGeometry(buf, hf, scene.map?.water ?? []) : null), [buf, hf, scene.map]);
  // Grass detail tiles every 4 m on the terrain and on the plain around it.
  const grass = useMemo(() => (hf ? grassTexture((2 * hf.halfX) / 4) : null), [hf]);
  const plainGrass = useMemo(() => (hf ? grassTexture(PLAIN / 4) : null), [hf]);
  const studio = useMemo(() => (hf ? null : studioTexture(PLAIN / 5)), [hf]);
  useEffect(() => () => terrain?.dispose(), [terrain]);
  useEffect(() => () => { grass?.dispose(); plainGrass?.dispose(); studio?.dispose(); }, [grass, plainGrass, studio]);
  if (terrain && hf && grass && plainGrass) {
    // The terrain fades to flat, plain grass at its edge; four strips of the same grass continue it
    // outwards (they don't overlap the terrain, so nothing fights for the same pixels).
    const s = hf.halfX, w = PLAIN / 2;
    const strips: [number, number, number, number][] = [
      [0, s + w / 2, 2 * (s + w), w], [0, -s - w / 2, 2 * (s + w), w],
      [s + w / 2, 0, w, 2 * s], [-s - w / 2, 0, w, 2 * s],
    ];
    return (
      <>
        {/* Distinct keys: without them React reuses the terrain mesh as the studio floor on a map switch,
            and dropping the geometry prop leaves the floor with an empty geometry (nothing drawn). */}
        <mesh key="terrain" geometry={terrain} receiveShadow>
          <meshStandardMaterial vertexColors map={grass} roughness={0.95} metalness={0} />
        </mesh>
        {strips.map(([x, y, sw, sh], i) => (
          <mesh key={`plain${i}`} position={[x, y, 0]} receiveShadow>
            <planeGeometry args={[sw, sh]} />
            <meshStandardMaterial color={GRASS_HEX} map={plainGrass} roughness={1} />
          </mesh>
        ))}
      </>
    );
  }
  return (
    <mesh key="studio-floor" position={[0, 0, -0.002]} receiveShadow>
      <planeGeometry args={[PLAIN, PLAIN]} />
      <meshStandardMaterial map={studio} roughness={0.85} metalness={0} />
    </mesh>
  );
}

// ---------------------------------------------------------------------------- markers
/** A ring in the body's colour under the selected body; a thin line up to it when airborne. */
function SelectionMarker() {
  const selected = useStore((s) => s.selected);
  const scene = useStore((s) => s.scene);
  const ring = useRef<THREE.Mesh>(null);
  const line = useRef<THREE.Mesh>(null);
  const root = scene?.agents.find((a) => a.key === selected)?.root ?? -1;
  const r = selected ? RING_RADIUS[selected] ?? 0.5 : 0.5;
  const color = useMemo(() => new THREE.Color(look(selected ?? "").color), [selected]);
  useFrame(() => {
    const c = poses.curr;
    if (!c || root < 0 || !ring.current || !line.current) return;
    const o = root * 7;
    const ground = c[o + 2] > 1.2 ? 0.04 : Math.min(c[o + 2] - 0.2, 0) + 0.04; // just above sidewalks (3 cm)
    ring.current.position.set(c[o], c[o + 1], ground);
    const h = Math.max(c[o + 2] - ground - 0.05, 0.001);
    line.current.visible = c[o + 2] > 0.6;
    line.current.position.set(c[o], c[o + 1], ground + h / 2);
    line.current.scale.set(1, 1, h);
  });
  if (!selected) return null;
  return (
    <>
      <mesh ref={ring} renderOrder={2}>
        <ringGeometry args={[r * 0.92, r, 96]} />
        <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.95} depthWrite={false} />
      </mesh>
      <mesh ref={line} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.004, 0.004, 1, 6]} />
        <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.6} />
      </mesh>
    </>
  );
}

/** Dotted white ring around the current goal target, and a flag where the car is heading. */
function GoalMarker() {
  const goal = useStore((s) => s.status?.goal ?? null);
  const apple = useStore((s) => s.status?.apple.position ?? null);
  const carTarget = useStore((s) => s.status?.bodies.car?.target ?? null);
  const ref = useRef<THREE.Group>(null);
  const point = goal?.target === "apple" ? apple : goal?.point ?? null;
  const dots = useMemo(() => Array.from({ length: 36 }, (_, i) => (i / 36) * Math.PI * 2), []);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.z += dt * 0.25;
  });
  return (
    <>
      {point && (
        <group ref={ref} position={[point[0], point[1], Math.max(point[2] - 0.03, 0.04)]}>
          {dots.map((t) => (
            <mesh key={t} position={[Math.cos(t) * 0.6, Math.sin(t) * 0.6, 0]}>
              <circleGeometry args={[0.025, 8]} />
              <meshBasicMaterial color={GOAL} toneMapped={false} />
            </mesh>
          ))}
        </group>
      )}
      {carTarget && (
        <group position={[carTarget[0], carTarget[1], 0]}>
          <mesh position={[0, 0, 0.6]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.02, 0.02, 1.2, 8]} />
            <meshBasicMaterial color="#ffffff" toneMapped={false} />
          </mesh>
          <mesh position={[0.18, 0, 1.08]}>
            <boxGeometry args={[0.34, 0.01, 0.22]} />
            <meshBasicMaterial color={look("car").color} toneMapped={false} />
          </mesh>
        </group>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------- picking
/** Click a body to select it; double-click to select and fly the camera to it. */
function Picker({ built }: { built: BuiltWorld }) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    const el = gl.domElement;
    const ray = new THREE.Raycaster();
    let down = { x: 0, y: 0, t: 0 };
    const pick = (e: MouseEvent): string | null => {
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObjects(built.agentGroups, true).find((h) => h.object.visible);
      return (hit?.object.userData.agent as string | undefined) ?? null;
    };
    const onDown = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY, t: performance.now() });
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6 || performance.now() - down.t > 400) return;
      const key = pick(e);
      if (key) setState({ selected: key, dock: "body" });
    };
    const onDbl = (e: MouseEvent) => {
      const key = pick(e);
      const s = getState();
      const root = s.scene?.agents.find((a) => a.key === key)?.root;
      if (!key || root === undefined || !poses.curr) return;
      setState({ selected: key, dock: "body" });
      const o = root * 7;
      flyTo(poses.curr[o], poses.curr[o + 1], key === "car" ? 7 : 4.5, poses.curr[o + 2]);
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("dblclick", onDbl);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("dblclick", onDbl);
    };
  }, [gl, camera, built]);
  return null;
}

// ---------------------------------------------------------------------------- camera
const held = new Set<string>();

function typingTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

/** True while the arrow keys belong to driving the car or flying the drone. */
export function arrowsDrive(): boolean {
  const s = getState();
  if (!s.selected || s.status?.bodies[s.selected]?.mode !== "manual") return false;
  const kind = s.scene?.agents.find((a) => a.key === s.selected)?.kind;
  return kind === "vehicle" || kind === "aerial";
}

/**
 * Keyboard + button navigation on top of OrbitControls:
 *   W A S D (or arrows)  move across the ground     Q / E  turn
 *   Z / X  (or - / =)    zoom                        R  back to the start view
 *   F  fly to the selected body    T  top view        follow keeps the selected body centred
 */
function CameraRig({ scene, buf }: { scene: SceneDef; buf: ArrayBuffer }) {
  const controls = useThree((s) => s.controls) as unknown as OrbitImpl | null;
  const camera = useThree((s) => s.camera);
  const topView = useStore((s) => s.topView);
  const frames = useRef({ n: 0, t: performance.now() });
  const dest = useRef<{ target: THREE.Vector3; distance: number } | null>(null);
  const tmp = useMemo(() => ({ a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3() }), []);

  // Start view for this scene.
  useEffect(() => {
    if (!controls) return;
    const v = homeView(scene);
    const t = toThree(v.x, v.y, scene.map?.key === "open" ? 0 : 0.7);
    controls.target.copy(t);
    const dir = scene.map?.key === "open" ? new THREE.Vector3(-0.45, 0.42, 0.78) : new THREE.Vector3(0, 0.3, 1);
    camera.position.copy(t).add(dir.normalize().multiplyScalar(v.distance));
    controls.update();
  }, [controls, camera, scene]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (typingTarget(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k.startsWith("arrow") && arrowsDrive()) return;
      if (["w", "a", "s", "d", "q", "e", "z", "x", "-", "=", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k)) {
        held.add(k);
        if (k.startsWith("arrow")) e.preventDefault();
      } else if (k === "r") {
        const v = homeView(scene);
        flyTo(v.x, v.y, v.distance);
        setState({ topView: false, follow: false });
      } else if (k === "t") {
        setState({ topView: !getState().topView });
      } else if (k === "f") {
        const s = getState();
        const root = s.scene?.agents.find((a) => a.key === s.selected)?.root;
        if (root !== undefined && poses.curr) {
          const o = root * 7;
          flyTo(poses.curr[o], poses.curr[o + 1], s.selected === "car" ? 7 : 4.5, poses.curr[o + 2]);
        }
      }
    };
    const up = (e: KeyboardEvent) => held.delete(e.key.toLowerCase());
    const blur = () => held.clear();
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, [scene]);

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05);
    const f = frames.current;
    f.n++;
    const now = performance.now();
    if (now - f.t > 500) {
      setState({ fps: Math.round((f.n * 1000) / (now - f.t)) });
      f.n = 0;
      f.t = now;
    }
    if (!controls) return;
    const target = controls.target;
    const offset = tmp.a.copy(camera.position).sub(target);
    let dist = offset.length();

    // Requests from the minimap, place buttons, double-click, F and R.
    if (cam.request) {
      const r = cam.request;
      // Places carry no height: aim at the ground there, not at a point buried inside a hill.
      dest.current = { target: toThree(r.x, r.y, r.z ?? groundAt(scene, buf, r.x, r.y)), distance: r.distance ?? dist };
      cam.request = null;
      setState({ follow: false });
    }
    // On-screen buttons and held keys.
    const turn = cam.nav.rotate + (held.has("q") ? 1 : 0) - (held.has("e") ? 1 : 0);
    const zoom = cam.nav.zoom + (held.has("x") || held.has("-") ? 1 : 0) - (held.has("z") || held.has("=") ? 1 : 0);
    const arrows = !arrowsDrive();
    const fwd = (held.has("w") || (arrows && held.has("arrowup")) ? 1 : 0) - (held.has("s") || (arrows && held.has("arrowdown")) ? 1 : 0);
    const side = (held.has("d") || (arrows && held.has("arrowright")) ? 1 : 0) - (held.has("a") || (arrows && held.has("arrowleft")) ? 1 : 0);

    if (turn) offset.applyAxisAngle(UP, turn * 1.6 * dt);
    if (zoom) {
      dist = THREE.MathUtils.clamp(dist * Math.exp(zoom * 1.4 * dt), 0.6, 140);
      offset.setLength(dist);
    }
    if (fwd || side) {
      const forward = tmp.b.set(-offset.x, 0, -offset.z).normalize();
      const right = tmp.c.set(-forward.z, 0, forward.x);
      const speed = Math.max(2, dist * 0.9) * dt;
      target.addScaledVector(forward, fwd * speed).addScaledVector(right, side * speed);
      dest.current = null;
      if (getState().follow) setState({ follow: false });
    }
    const s = getState();
    if (s.follow && s.selected && poses.curr && s.scene) {
      const root = s.scene.agents.find((a) => a.key === s.selected)?.root;
      if (root !== undefined) {
        const o = root * 7;
        target.lerp(toThree(poses.curr[o], poses.curr[o + 1], poses.curr[o + 2], tmp.b), 1 - Math.exp(-dt * 5));
      }
    }
    if (dest.current) {
      const k = 1 - Math.exp(-dt * 5);
      target.lerp(dest.current.target, k);
      dist += (dest.current.distance - dist) * k;
      offset.setLength(dist);
      if (target.distanceTo(dest.current.target) < 0.02 && Math.abs(dist - dest.current.distance) < 0.02) dest.current = null;
    }
    // Top view: swing the camera straight above the target (and back down when turned off).
    const flat = Math.hypot(offset.x, offset.z);
    if (topView && flat > 0.03 * dist) {
      const len = offset.length();
      const heading = Math.atan2(offset.x, offset.z);
      const newFlat = flat * Math.exp(-dt * 5);
      offset.set(Math.sin(heading) * newFlat, Math.sqrt(Math.max(len * len - newFlat * newFlat, 1e-4)), Math.cos(heading) * newFlat);
    } else if (!topView && flat < 0.5 * dist) {
      const len = offset.length();
      const heading = Math.atan2(offset.x, offset.z) || 0;
      const newFlat = flat + (0.7 * dist - flat) * (1 - Math.exp(-dt * 5));
      offset.set(Math.sin(heading) * newFlat, Math.sqrt(Math.max(len * len - newFlat * newFlat, 1e-4)), Math.cos(heading) * newFlat);
    }
    // Never go underground: keep the camera at least 1.5 m above the terrain below it.
    const floor = groundAt(scene, buf, target.x + offset.x, -(target.z + offset.z)) + 1.5;
    if (target.y + offset.y < floor) offset.y = floor - target.y;
    camera.position.copy(target).add(offset);

    // Share with the minimap (MuJoCo coordinates).
    cam.x = target.x;
    cam.y = -target.z;
    cam.yaw = Math.atan2(offset.z, -offset.x);
    cam.distance = dist;
  });
  return null;
}

/** Sun that follows the camera target, so shadows work anywhere in the big world. */
function Sun() {
  const light = useRef<THREE.DirectionalLight>(null);
  const controls = useThree((s) => s.controls) as unknown as OrbitImpl | null;
  useFrame(() => {
    const l = light.current;
    if (!l || !controls) return;
    const t = controls.target;
    l.position.set(t.x + 16, t.y + 40, t.z + 12);
    l.target.position.copy(t);
    l.target.updateMatrixWorld();
    const half = THREE.MathUtils.clamp(cam.distance * 1.2, 20, 90);
    const sc = l.shadow.camera;
    if (Math.abs(sc.right - half) > half * 0.1) {
      sc.left = sc.bottom = -half;
      sc.right = sc.top = half;
      sc.updateProjectionMatrix();
    }
  });
  return (
    <directionalLight
      ref={light}
      castShadow
      intensity={2.4}
      color="#fff6e8"
      shadow-mapSize={[4096, 4096]}
      shadow-bias={-0.0003}
      shadow-normalBias={0.02}
      shadow-camera-left={-30}
      shadow-camera-right={30}
      shadow-camera-top={30}
      shadow-camera-bottom={-30}
      shadow-camera-far={120}
    />
  );
}

export default function World() {
  const scene = useStore((s) => s.scene);
  const meshes = useStore((s) => s.meshes);
  const built = useMemo(() => (scene && meshes ? buildWorld(scene, meshes) : null), [scene, meshes]);
  useEffect(() => () => built?.dispose(), [built]);
  if (!scene || !meshes) return null;

  return (
    <Canvas
      shadows="percentage"
      dpr={[1, 2]}
      camera={{ position: [0, 3, 8], fov: 40, near: 0.03, far: 3000 }}
      // Logarithmic depth: roads, sidewalks and rugs sit millimetres to centimetres apart, and a normal
      // depth buffer can't separate them beyond ~40 m (the road shredded into the grass when zoomed out).
      gl={{ antialias: true, powerPreference: "high-performance", logarithmicDepthBuffer: true }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.0;
      }}
    >
      <Environment files="/hdri/sky_2k.hdr" background environmentIntensity={0.85} />
      <fog attach="fog" args={[HAZE, scene.map?.key === "open" ? 80 : 40, scene.map?.key === "open" ? 520 : 260]} />
      <Sun />
      <MujocoFrame>
        <Ground scene={scene} buf={meshes} />
        {built && <Bodies built={built} />}
        {built && <SelectionMarker />}
        <GoalMarker />
      </MujocoFrame>
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} screenSpacePanning={false}
        maxPolarAngle={Math.PI * 0.49} minDistance={0.6} maxDistance={140} />
      <CameraRig scene={scene} buf={meshes} />
      {built && <Picker built={built} />}
    </Canvas>
  );
}
