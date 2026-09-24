import { useState } from "react";
import { Brain, CircleHelp, Pause, Play, Plus, RotateCcw, SlidersHorizontal, X } from "lucide-react";
import { api, reportError, switchScene } from "../lib/api";
import { getState, setState, useStore } from "../lib/store";
import { condition, look } from "./theme";

function StatusPill() {
  const link = useStore((s) => s.link);
  const paused = useStore((s) => s.status?.paused ?? false);
  const rtf = useStore((s) => s.status?.rtf ?? 0);
  const state = link !== "open" ? (link === "closed" ? "offline" : "connecting") : paused ? "paused" : "live";
  const label = { live: "Live", paused: "Paused", offline: "Offline", connecting: "Connecting" }[state];
  return (
    <span className={`pill ${state}`} title={state === "live" ? `Simulation running at ${rtf.toFixed(2)}× real time` : undefined}>
      <i aria-hidden="true" />
      {label}
    </span>
  );
}

function MapSwitch() {
  const config = useStore((s) => s.config);
  const switching = useStore((s) => s.switching);
  if (!config) return null;
  return (
    <div className="segmented map-switch" role="radiogroup" aria-label="World">
      {config.maps.map((m) => (
        <button key={m.key} type="button" role="radio" aria-checked={config.map === m.key}
          className={config.map === m.key ? "is-on" : ""} disabled={!!switching}
          onClick={() => config.map !== m.key && switchScene(m.key, config.bodies, `Opening the ${m.label.toLowerCase()}…`)}>
          {m.label}
        </button>
      ))}
    </div>
  );
}

/** The bodies in this scene: select to inspect, × to remove, + to add one more. */
function Lineup() {
  const config = useStore((s) => s.config);
  const scene = useStore((s) => s.scene);
  const status = useStore((s) => s.status);
  const selected = useStore((s) => s.selected);
  const switching = useStore((s) => s.switching);
  const [adding, setAdding] = useState(false);
  if (!config || !scene) return null;
  const inScene = config.bodies;
  const others = config.all.filter((b) => config.available.includes(b.key) && !inScene.includes(b.key));
  const change = (bodies: string[], msg: string) => {
    setAdding(false);
    void switchScene(config.map, config.all.map((b) => b.key).filter((k) => bodies.includes(k)), msg);
  };

  return (
    <nav className="chips" aria-label="Bodies in this scene">
      {scene.agents.map((a) => {
        const L = look(a.key);
        const cond = condition(a, status?.bodies[a.key]);
        const on = selected === a.key;
        const Icon = L.icon;
        return (
          <span key={a.key} className={`chip ${on ? "is-on" : ""}`} style={{ "--c": L.color } as React.CSSProperties}>
            <button type="button" className="chip-main" aria-pressed={on}
              onClick={() => setState(on ? { selected: null } : { selected: a.key, dock: "body" })}>
              <Icon size={16} aria-hidden="true" />
              <span className="chip-name">{a.label}</span>
              <span className={`chip-state ${cond.bad ? "bad" : ""}`}>{cond.word}</span>
            </button>
            {inScene.length > 1 && (
              <button type="button" className="chip-x" aria-label={`Remove ${a.label} from the scene`} disabled={!!switching}
                onClick={() => change(inScene.filter((k) => k !== a.key), `Removing the ${a.label.toLowerCase()}…`)}>
                <X size={13} />
              </button>
            )}
          </span>
        );
      })}
      {others.length > 0 && (
        <span className="add-wrap">
          <button type="button" className="chip add" aria-expanded={adding} disabled={!!switching} onClick={() => setAdding(!adding)}>
            <Plus size={16} aria-hidden="true" />
            <span>Add body</span>
          </button>
          {adding && (
            <div className="menu" role="menu">
              {others.map((b) => {
                const L = look(b.key);
                const Icon = L.icon;
                return (
                  <button key={b.key} type="button" role="menuitem" style={{ "--c": L.color } as React.CSSProperties}
                    onClick={() => change([...inScene, b.key], `Adding the ${b.label.toLowerCase()}…`)}>
                    <Icon size={16} aria-hidden="true" /> {b.label}
                  </button>
                );
              })}
              {others.length > 1 && (
                <button type="button" role="menuitem" className="all"
                  onClick={() => change(config.available, "Adding every body…")}>
                  Add all {others.length}
                </button>
              )}
            </div>
          )}
        </span>
      )}
    </nav>
  );
}

export default function TopBar() {
  const paused = useStore((s) => s.status?.paused ?? false);
  const dock = useStore((s) => s.dock);
  const menu = useStore((s) => s.worldMenu);
  const live = useStore((s) => s.link === "open");
  return (
    <header className="topbar">
      <div className="topbar-row">
        <div className="brand-row">
          <div className="brand">
            <span className="brand-name">Bio-Bot Studio</span>
            <StatusPill />
          </div>
          <MapSwitch />
        </div>
        <div className="actions">
          <button type="button" className="btn ghost" aria-label={paused ? "Resume" : "Pause"} disabled={!live}
            onClick={() => api.pause(!paused).catch(reportError)}>
            {paused ? <Play size={16} /> : <Pause size={16} />}
            <span>{paused ? "Resume" : "Pause"}</span>
          </button>
          <button type="button" className="btn ghost" aria-label="Reset the world" disabled={!live}
            onClick={() => api.resetWorld().catch(reportError)}>
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
          <button type="button" className={`btn ghost ${menu ? "is-on" : ""}`} aria-label="World settings" aria-expanded={menu}
            onClick={() => setState({ worldMenu: !menu, help: false })}>
            <SlidersHorizontal size={16} />
            <span>World</span>
          </button>
          <button type="button" className={`btn ${dock === "brain" ? "primary" : "ghost"}`} aria-label="Brain" aria-pressed={dock === "brain"}
            onClick={() => setState({ dock: dock === "brain" ? (getState().selected ? "body" : null) : "brain" })}>
            <Brain size={16} />
            <span>Brain</span>
          </button>
          <button type="button" className="btn ghost icon-only" aria-label="Keyboard and mouse help"
            onClick={() => setState({ help: !getState().help })}>
            <CircleHelp size={16} />
          </button>
        </div>
      </div>
      <Lineup />
    </header>
  );
}
