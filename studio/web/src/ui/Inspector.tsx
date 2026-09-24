import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Camera, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Crosshair, RotateCcw } from "lucide-react";
import { api, connectInspect, reportError } from "../lib/api";
import { camera, flyTo, getState, poses, setState, useStore } from "../lib/store";
import type { AgentDef, BodyStatus, InspectInfo, LiveInfo, Mode } from "../lib/types";
import { condition, look, modeHelp, modeLabel, pretty } from "./theme";

const MODES: Mode[] = ["hold", "off", "babble", "manual"];
const deg = (r: number) => `${Math.round((r * 180) / Math.PI)}°`;

/** Arrow keys drive the car or fly the drone while that body is in manual mode. */
function useArrowControl(agent: AgentDef, manual: boolean) {
  useEffect(() => {
    if (!manual || (agent.kind !== "vehicle" && agent.kind !== "aerial")) return;
    const keys = new Set<string>();
    const typing = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
    };
    const down = (e: KeyboardEvent) => {
      if (typing(e) || !e.key.startsWith("Arrow")) return;
      keys.add(e.shiftKey ? `shift+${e.key}` : e.key);
      e.preventDefault();
    };
    const up = (e: KeyboardEvent) => {
      keys.delete(e.key);
      keys.delete(`shift+${e.key}`);
    };
    const tick = window.setInterval(() => {
      const on = (k: string) => keys.has(k);
      if (agent.kind === "vehicle") {
        const throttle = (on("ArrowUp") ? 1 : 0) - (on("ArrowDown") ? 1 : 0);
        const steer = (on("ArrowLeft") ? 1 : 0) - (on("ArrowRight") ? 1 : 0);
        if (keys.size) api.drive(agent.key, throttle, steer).catch(reportError);
      } else {
        const f = (on("ArrowUp") ? 1 : 0) - (on("ArrowDown") ? 1 : 0);
        const r = (on("ArrowRight") ? 1 : 0) - (on("ArrowLeft") ? 1 : 0);
        const z = (on("shift+ArrowUp") ? 1 : 0) - (on("shift+ArrowDown") ? 1 : 0);
        if (f || r || z) {
          // Move relative to where the camera looks, so "up" always means "away from me".
          const cx = Math.cos(camera.yaw), cy = Math.sin(camera.yaw);
          api.nudge(agent.key, 0.25 * (f * cx + r * cy), 0.25 * (f * cy - r * cx), 0.2 * z).catch(reportError);
        }
      }
    }, 100);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.clearInterval(tick);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [agent, manual]);
}

/** Press-and-hold button that repeats an action every 100 ms. */
function HoldButton({ label, onHold, children }: { label: string; onHold: () => void; children: React.ReactNode }) {
  const timer = useRef(0);
  const stop = () => window.clearInterval(timer.current);
  useEffect(() => stop, []);
  return (
    <button type="button" className="pad-btn" aria-label={label}
      onPointerDown={(e) => { e.preventDefault(); onHold(); timer.current = window.setInterval(onHold, 100); }}
      onPointerUp={stop} onPointerLeave={stop} onPointerCancel={stop}>
      {children}
    </button>
  );
}

function Pad({ agent, live }: { agent: AgentDef; live: LiveInfo | null }) {
  const places = useStore((s) => s.scene?.map?.places ?? []);
  const vehicle = agent.kind === "vehicle";
  const drive = (t: number, s: number) => api.drive(agent.key, t, s).catch(reportError);
  const nudge = (f: number, r: number, z: number) => {
    const cx = Math.cos(camera.yaw), cy = Math.sin(camera.yaw);
    api.nudge(agent.key, 0.3 * (f * cx + r * cy), 0.3 * (f * cy - r * cx), 0.25 * z).catch(reportError);
  };
  return (
    <div className="section">
      <div className="pad">
        <span />
        <HoldButton label={vehicle ? "Accelerate" : "Forward"} onHold={() => (vehicle ? drive(1, 0) : nudge(1, 0, 0))}><ChevronUp size={18} /></HoldButton>
        <span />
        <HoldButton label={vehicle ? "Steer left" : "Left"} onHold={() => (vehicle ? drive(0.6, 1) : nudge(0, -1, 0))}><ChevronLeft size={18} /></HoldButton>
        <span className="pad-mid">{vehicle ? `${(live?.drive?.speed ?? 0).toFixed(1)} m/s` : `${(live?.setpoint?.[2] ?? 0).toFixed(1)} m`}</span>
        <HoldButton label={vehicle ? "Steer right" : "Right"} onHold={() => (vehicle ? drive(0.6, -1) : nudge(0, 1, 0))}><ChevronRight size={18} /></HoldButton>
        <span />
        <HoldButton label={vehicle ? "Reverse" : "Back"} onHold={() => (vehicle ? drive(-1, 0) : nudge(-1, 0, 0))}><ChevronDown size={18} /></HoldButton>
        <span />
      </div>
      {!vehicle && (
        <div className="row-actions">
          <HoldButton label="Climb" onHold={() => nudge(0, 0, 1)}><ArrowUp size={16} /> Up</HoldButton>
          <HoldButton label="Descend" onHold={() => nudge(0, 0, -1)}><ArrowDown size={16} /> Down</HoldButton>
        </div>
      )}
      <p className="muted small">
        Keyboard: arrow keys {vehicle ? "drive and steer" : "move it (relative to your view); Shift + ↑/↓ changes height"}.
      </p>
      {places.length > 1 && (
        <>
          <h3 className="section-title">{vehicle ? "Drive itself to" : "Fly itself to"}</h3>
          <div className="place-list">
            {places.map((p) => (
              <button key={p.key} type="button" className="suggestion" onClick={() => api.goto(agent.key, p.key).catch(reportError)}>
                {p.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Joints({ info, live, agent }: { info: InspectInfo; live: LiveInfo | null; agent: AgentDef }) {
  const [filter, setFilter] = useState("");
  const [drag, setDrag] = useState<{ i: number; v: number } | null>(null);
  const last = useRef(0);
  const manual = live?.mode === "manual";
  const rows = info.actuators.filter((a) => pretty(a.name).toLowerCase().includes(filter.toLowerCase()));
  const send = (i: number, v: number, force = false) => {
    const now = performance.now();
    if (!force && now - last.current < 60) return;
    last.current = now;
    api.manual(agent.key, i, v).catch(reportError);
  };
  return (
    <div className="section">
      <div className="section-head">
        <h3 className="section-title">Motors <span className="count">{info.actuators.length}</span></h3>
        {info.actuators.length > 12 && (
          <input className="filter" placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter motors" />
        )}
      </div>
      {!manual && <p className="muted small">Live values. Pick <strong>Take control</strong> above to move them yourself.</p>}
      <ul className="joints" onMouseLeave={() => setState({ hoverPart: null })}>
        {rows.map((a) => {
          const value = drag?.i === a.index ? drag.v : live?.manual[a.index] ?? 0;
          const angle = a.unit === "rad";
          const current = angle ? live?.joints[info.joints.findIndex((j) => j.name === a.joint)] : live?.ctrl[a.index];
          return (
            <li key={a.index} onMouseEnter={() => setState({ hoverPart: a.part })}>
              <div className="joint-head">
                <span>{pretty(a.name)}</span>
                <span className="muted small tabular">
                  {current === undefined ? "–" : angle ? deg(current) : `${current.toFixed(2)} ${a.unit}`}
                </span>
              </div>
              <input type="range" min={a.range[0]} max={a.range[1]} step={(a.range[1] - a.range[0]) / 200}
                value={value} disabled={!manual} aria-label={`${pretty(a.name)} target`}
                onChange={(e) => { const v = Number(e.target.value); setDrag({ i: a.index, v }); send(a.index, v); }}
                onPointerUp={() => { if (drag) send(drag.i, drag.v, true); setDrag(null); }}
                onKeyUp={() => { if (drag) send(drag.i, drag.v, true); setDrag(null); }} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Sensors({ info, live }: { info: InspectInfo; live: LiveInfo | null }) {
  if (!info.sensors.length) return null;
  let at = 0;
  return (
    <details className="section">
      <summary className="section-title">Sensors <span className="count">{info.sensors.length}</span></summary>
      <ul className="sensors">
        {info.sensors.map((s) => {
          const vals = live?.sensors.slice(at, at + s.dim) ?? [];
          at += s.dim;
          return (
            <li key={s.name}>
              <span>{pretty(s.name)} <span className="muted small">({s.type})</span></span>
              <span className="tabular small">{vals.map((v) => v.toFixed(2)).join("  ") || "–"}</span>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

export default function Inspector() {
  const key = useStore((s) => s.selected);
  const scene = useStore((s) => s.scene);
  const b = useStore((s) => (s.selected ? s.status?.bodies[s.selected] : undefined));
  const info = useStore((s) => s.inspect);
  const live = useStore((s) => s.live);
  const follow = useStore((s) => s.follow);
  const agent = useMemo(() => scene?.agents.find((x) => x.key === key), [scene, key]);
  const [power, setPower] = useState<number | null>(null);

  useEffect(() => {
    if (!key || !scene) return;
    setState({ inspect: null, live: null });
    api.inspect(key).then((i) => getState().selected === key && setState({ inspect: i })).catch(reportError);
    const stop = connectInspect(key);
    return () => { stop(); setState({ hoverPart: null }); };
  }, [key, scene]);
  useEffect(() => setPower(null), [key]);

  if (!agent || !key) return <p className="muted pad-text">Click a body in the world or in the bar at the top to inspect it.</p>;
  const L = look(key);
  const Icon = L.icon;
  const cond = condition(agent, b);
  const mode = b?.mode ?? "hold";
  const shown = power ?? Math.round((b?.strength ?? 1) * 100);
  const focus = () => {
    if (!poses.curr) return;
    const o = agent.root * 7;
    flyTo(poses.curr[o], poses.curr[o + 1], key === "car" ? 7 : 4.5, poses.curr[o + 2]);
  };

  return <InspectorBody {...{ agent, L, Icon, cond, mode, shown, focus, follow, info, live, b, setPower, power }} />;
}

function InspectorBody(p: {
  agent: AgentDef; L: ReturnType<typeof look>; Icon: ReturnType<typeof look>["icon"]; cond: { word: string; bad: boolean };
  mode: Mode; shown: number; focus: () => void; follow: boolean; info: InspectInfo | null; live: LiveInfo | null;
  b: BodyStatus | undefined;
  setPower: (v: number | null) => void; power: number | null;
}) {
  const { agent, L, Icon, cond, mode, shown, focus, follow, info, live, b, setPower, power } = p;
  const manual = mode === "manual";
  useArrowControl(agent, manual);
  const commitPower = () => power !== null && api.setStrength(agent.key, power / 100).catch(reportError).finally(() => setPower(null));
  return (
    <div className="inspector" style={{ "--c": L.color } as React.CSSProperties}>
      <header className="card-head">
        <span className="badge"><Icon size={18} aria-hidden="true" /></span>
        <div>
          <h2>{agent.label}</h2>
          <p className="muted small">{L.model}</p>
        </div>
      </header>
      <p className="body-status">
        <span className={`state ${cond.bad ? "bad" : ""}`}>{cond.word}</span>
        {b && <span className="muted"> · {b.position[2].toFixed(2)} m up · {b.speed.toFixed(1)} m/s</span>}
      </p>

      <div className="segmented four" role="radiogroup" aria-label={`${agent.label} behaviour`}>
        {MODES.map((m) => (
          <button key={m} type="button" role="radio" aria-checked={mode === m} className={mode === m ? "is-on" : ""}
            onClick={() => api.setMode(agent.key, m).catch(reportError)}>
            {modeLabel(agent.kind, m)}
          </button>
        ))}
      </div>
      <p className="muted small help">{modeHelp(agent.kind, mode)}</p>

      {(agent.kind === "vehicle" || agent.kind === "aerial") && (manual || agent.kind === "vehicle" || mode === "hold") && (
        <Pad agent={agent} live={live} />
      )}

      <label className="slider">
        <span className="slider-head"><span>Motor power</span><strong>{shown}%</strong></span>
        <input type="range" min={0} max={100} step={5} value={shown}
          onChange={(e) => setPower(Number(e.target.value))} onPointerUp={commitPower} onKeyUp={commitPower} />
        <span className="muted small">Lower it to simulate a weak or damaged motor.</span>
      </label>

      <div className="row-actions">
        <button type="button" className="btn" title="Return it to where it started"
          onClick={() => api.resetBody(agent.key)
            .then(() => (b?.mode === "hold" || agent.kind !== "legged" ? undefined : api.setMode(agent.key, "hold")))
            .catch(reportError)}>
          <RotateCcw size={16} /> Put back
        </button>
        <button type="button" className="btn" onClick={focus}><Crosshair size={16} /> Find</button>
        <button type="button" className={`btn ${follow ? "primary" : ""}`} aria-pressed={follow}
          onClick={() => setState({ follow: !follow })}>
          <Camera size={16} /> {follow ? "Following" : "Follow"}
        </button>
      </div>

      {info && agent.kind !== "vehicle" && agent.kind !== "aerial" && <Joints info={info} live={live} agent={agent} />}
      {info && <Sensors info={info} live={live} />}
      {info && (
        <p className="muted small facts">
          {info.mass.toFixed(1)} kg · {info.parts.length} parts · {info.joints.length} joints · {info.actuators.length} motors
        </p>
      )}
      <p className="credit">{agent.credit}</p>
    </div>
  );
}
