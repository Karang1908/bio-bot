import { useEffect, useRef, useState } from "react";
import { House, Map as MapIcon, Minus, Plus, RotateCcw, RotateCw, ScanEye, X } from "lucide-react";
import { camera, flyTo, getState, poses, setState, useStore } from "../lib/store";
import type { SceneDef } from "../lib/types";
import { homeView } from "../world/views";
import { look } from "./theme";

const PX = 208; // minimap size in CSS pixels

/** Terrain, water and roads drawn once per scene into an offscreen canvas. */
function paintBase(scene: SceneDef, meshes: ArrayBuffer, half: number, dpr: number): HTMLCanvasElement {
  const size = Math.round(PX * dpr);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const s = size / (2 * half); // pixels per metre
  const toPx = (x: number, y: number): [number, number] => [(x + half) * s, (half - y) * s];

  const hf = scene.hfields[0];
  if (hf) {
    const h = new Float32Array(meshes, hf.offset, hf.nrow * hf.ncol);
    const img = g.createImageData(size, size);
    for (let py = 0; py < size; py++) {
      const y = half - (py + 0.5) / s;
      const row = Math.round(((y + hf.halfY) / (2 * hf.halfY)) * (hf.nrow - 1));
      for (let px = 0; px < size; px++) {
        const x = (px + 0.5) / s - half;
        const col = Math.round(((x + hf.halfX) / (2 * hf.halfX)) * (hf.ncol - 1));
        const inside = row >= 0 && row < hf.nrow && col >= 0 && col < hf.ncol;
        const z = inside ? h[row * hf.ncol + col] : 0;
        const k = (py * size + px) * 4;
        const t = Math.max(0, Math.min(1, z / 6));
        img.data[k] = 58 + 70 * t;
        img.data[k + 1] = 82 + 44 * t;
        img.data[k + 2] = 44 + 50 * t;
        img.data[k + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
  } else {
    g.fillStyle = "#22252c";
    g.fillRect(0, 0, size, size);
    g.strokeStyle = "rgba(255,255,255,0.06)";
    g.lineWidth = 1;
    for (let m = -half; m <= half; m += 5) {
      const [a] = toPx(m, 0);
      g.beginPath(); g.moveTo(a, 0); g.lineTo(a, size); g.stroke();
      g.beginPath(); g.moveTo(0, a); g.lineTo(size, a); g.stroke();
    }
  }

  for (const w of scene.map?.water ?? []) {
    const [cx, cy] = toPx(w.cx, w.cy);
    g.fillStyle = "#3b7bb8";
    g.beginPath(); g.ellipse(cx, cy, w.rx * s, w.ry * s, 0, 0, Math.PI * 2); g.fill();
  }

  // Static scenery (roads, house, furniture, stairs, tree trunks) at its live pose. Free objects
  // like the apple and balls move, so they're left out; the apple gets its own marker.
  const q = poses.curr;
  const moving = new Set(["apple", "ball0", "ball1", "ball2"]);
  const bodyName = new Map(scene.bodies.map((b) => [b.id, b.name]));
  const geoms = scene.geoms.filter((geom) => !geom.agent && (geom.type === "box" || geom.type === "cylinder")
    && !moving.has(bodyName.get(geom.body) ?? "") && !geom.name.startsWith("lamp"));
  geoms.sort((a, b) => Number(!a.name.startsWith("road")) - Number(!b.name.startsWith("road")));
  for (const geom of geoms) {
    let bx = 0, by = 0, byaw = 0;
    if (geom.body !== 0 && q) {
      const o = geom.body * 7;
      bx = q[o]; by = q[o + 1]; byaw = yawOf(q, o);
    }
    const [q0, q1, q2, q3] = geom.quat;
    const gyaw = Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
    const c0 = Math.cos(byaw), s0 = Math.sin(byaw);
    const [x, y] = toPx(bx + c0 * geom.pos[0] - s0 * geom.pos[1], by + s0 * geom.pos[0] + c0 * geom.pos[1]);
    g.save();
    g.translate(x, y);
    g.rotate(-(byaw + gyaw));
    if (geom.type === "box") {
      g.fillStyle = geom.name.startsWith("road") ? "#4b4d52" : geom.name.startsWith("sidewalk") ? "#8d8b86"
        : geom.name.startsWith("bldg") ? "#c9ced6" : "rgba(235,228,214,0.6)";
      g.fillRect(-geom.size[0] * s, -geom.size[1] * s, 2 * geom.size[0] * s, 2 * geom.size[1] * s);
    } else {
      g.fillStyle = "#2a4220";
      g.beginPath(); g.arc(0, 0, Math.max(geom.size[0] * s * 4, 2 * dpr), 0, Math.PI * 2); g.fill();
    }
    g.restore();
  }
  return c;
}

function yawOf(q: Float32Array, o: number): number {
  const w = q[o + 3], x = q[o + 4], y = q[o + 5], z = q[o + 6];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function Minimap({ scene, meshes }: { scene: SceneDef; meshes: ArrayBuffer }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const half = scene.map?.size ?? 20;

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = el.height = Math.round(PX * dpr);
    let base = paintBase(scene, meshes, half, dpr);
    let posed = !!poses.curr;
    const g = el.getContext("2d")!;
    const s = el.width / (2 * half);
    const toPx = (x: number, y: number): [number, number] => [(x + half) * s, (half - y) * s];
    let raf = 0;
    let last = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (now - last < 66) return; // ~15 fps is plenty for a map
      last = now;
      const st = getState();
      if (!posed && poses.curr) {
        base = paintBase(scene, meshes, half, dpr);
        posed = true;
      }
      g.drawImage(base, 0, 0);

      g.font = `${11 * dpr}px system-ui, sans-serif`;
      g.textAlign = "center";
      for (const p of scene.map?.places ?? []) {
        if ((scene.map?.places.length ?? 0) < 2) break;
        const [x, y] = toPx(p.x, p.y);
        g.fillStyle = "rgba(0,0,0,0.55)";
        g.fillText(p.label, x + dpr, y + dpr);
        g.fillStyle = "rgba(255,255,255,0.92)";
        g.fillText(p.label, x, y);
      }

      const apple = st.status?.apple.position;
      if (apple) {
        const [x, y] = toPx(apple[0], apple[1]);
        g.fillStyle = "#ef4444";
        g.strokeStyle = "#fff";
        g.lineWidth = 1.5 * dpr;
        g.beginPath(); g.arc(x, y, 3.5 * dpr, 0, Math.PI * 2); g.fill(); g.stroke();
      }

      // The camera: where it's looking and roughly how much of the world is in view.
      const [cx, cy] = toPx(camera.x, camera.y);
      const reach = Math.max(18 * dpr, Math.min(camera.distance * 1.6 * s, el.width));
      g.fillStyle = "rgba(255,255,255,0.16)";
      g.beginPath();
      g.moveTo(cx, cy);
      g.arc(cx, cy, reach, -camera.yaw - 0.45, -camera.yaw + 0.45);
      g.closePath();
      g.fill();
      g.strokeStyle = "#fff";
      g.lineWidth = 1.5 * dpr;
      g.beginPath(); g.arc(cx, cy, 4 * dpr, 0, Math.PI * 2); g.stroke();

      const q = poses.curr;
      if (q) {
        for (const a of scene.agents) {
          const o = a.root * 7;
          const [x, y] = toPx(q[o], q[o + 1]);
          const yaw = yawOf(q, o);
          const on = st.selected === a.key;
          const r = (on ? 6 : 4.5) * dpr;
          g.fillStyle = look(a.key).color;
          g.strokeStyle = on ? "#fff" : "rgba(0,0,0,0.6)";
          g.lineWidth = (on ? 2 : 1.2) * dpr;
          if (a.kind === "vehicle") {
            g.save();
            g.translate(x, y);
            g.rotate(-yaw);
            g.beginPath(); g.roundRect(-r * 1.3, -r * 0.8, r * 2.6, r * 1.6, 2 * dpr); g.fill(); g.stroke();
            g.restore();
            continue;
          }
          g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill(); g.stroke();
          g.beginPath();
          g.moveTo(x + Math.cos(yaw) * r * 0.2, y - Math.sin(yaw) * r * 0.2);
          g.lineTo(x + Math.cos(yaw) * r * 2, y - Math.sin(yaw) * r * 2);
          g.stroke();
        }
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [scene, meshes, half]);

  // Click or drag on the map moves the camera there; clicking a body dot selects it too.
  const toWorld = (e: React.PointerEvent) => {
    const r = e.currentTarget.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * 2 * half - half, half - ((e.clientY - r.top) / r.height) * 2 * half];
  };
  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const [x, y] = toWorld(e);
    const q = poses.curr;
    const hit = q && scene.agents.find((a) => Math.hypot(q[a.root * 7] - x, q[a.root * 7 + 1] - y) < half * 0.05);
    if (hit) {
      setState({ selected: hit.key, dock: "body" });
      flyTo(q[hit.root * 7], q[hit.root * 7 + 1], hit.key === "car" ? 7 : 4.5, q[hit.root * 7 + 2]);
    } else {
      flyTo(x, y);
    }
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    const [x, y] = toWorld(e);
    flyTo(x, y);
  };

  return (
    <canvas ref={canvas} className="minimap" style={{ width: PX, height: PX }}
      onPointerDown={onDown} onPointerMove={onMove}
      role="img" aria-label="Map of the world. Click to move the camera there; click a coloured dot to select that body." />
  );
}

/** Button that applies a camera motion for as long as it's held. */
function NavHold({ label, set, children }: { label: string; set: (v: number) => void; children: React.ReactNode }) {
  return (
    <button type="button" className="nav-btn" aria-label={label} title={label}
      onPointerDown={(e) => { e.preventDefault(); set(1); }}
      onPointerUp={() => set(0)} onPointerLeave={() => set(0)} onPointerCancel={() => set(0)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") set(1); }}
      onKeyUp={() => set(0)} onBlur={() => set(0)}>
      {children}
    </button>
  );
}

/** Bottom-left: a live map you can click, camera buttons, and one-click trips to each place. */
export default function Navigator() {
  const scene = useStore((s) => s.scene);
  const meshes = useStore((s) => s.meshes);
  const topView = useStore((s) => s.topView);
  const [open, setOpen] = useState(() => window.matchMedia("(min-width: 821px)").matches);
  if (!scene || !meshes) return null;
  const places = scene.map?.places ?? [];

  if (!open) {
    return (
      <button type="button" className="btn ghost nav-open" onClick={() => setOpen(true)} aria-label="Show the map">
        <MapIcon size={16} /> <span>Map</span>
      </button>
    );
  }
  return (
    <section className="navigator" aria-label="Map and camera">
      <div className="nav-head">
        <strong>{scene.map?.label ?? "Map"}</strong>
        <button type="button" className="icon-btn" aria-label="Hide the map" onClick={() => setOpen(false)}><X size={16} /></button>
      </div>
      <Minimap scene={scene} meshes={meshes} />
      <div className="nav-row" role="toolbar" aria-label="Camera">
        <NavHold label="Turn left (Q)" set={(v) => (camera.nav.rotate = v)}><RotateCcw size={16} /></NavHold>
        <NavHold label="Turn right (E)" set={(v) => (camera.nav.rotate = -v)}><RotateCw size={16} /></NavHold>
        <NavHold label="Zoom in (Z)" set={(v) => (camera.nav.zoom = -v)}><Plus size={16} /></NavHold>
        <NavHold label="Zoom out (X)" set={(v) => (camera.nav.zoom = v)}><Minus size={16} /></NavHold>
        <button type="button" className={`nav-btn ${topView ? "is-on" : ""}`} aria-pressed={topView} title="Top view (T)"
          aria-label="Top view (T)" onClick={() => setState({ topView: !topView })}><ScanEye size={16} /></button>
        <button type="button" className="nav-btn" title="Start view (R)" aria-label="Start view (R)"
          onClick={() => { const v = homeView(scene); flyTo(v.x, v.y, v.distance); setState({ topView: false }); }}>
          <House size={16} />
        </button>
      </div>
      {places.length > 1 && (
        <div className="place-list" aria-label="Go to a place">
          {places.map((p) => (
            <button key={p.key} type="button" className="suggestion" onClick={() => flyTo(p.x, p.y, p.view)}>{p.label}</button>
          ))}
        </div>
      )}
    </section>
  );
}
