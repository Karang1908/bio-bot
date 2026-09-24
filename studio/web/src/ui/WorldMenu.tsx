import { X } from "lucide-react";
import { api, reportError, switchScene } from "../lib/api";
import { setState, useStore } from "../lib/store";

const SPEEDS = [0.5, 1, 2];

function sentence(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function WorldMenu() {
  const status = useStore((s) => s.status);
  const map = useStore((s) => s.scene?.map ?? null);
  const topView = useStore((s) => s.topView);
  const showCollision = useStore((s) => s.showCollision);
  const fps = useStore((s) => s.fps);
  const config = useStore((s) => s.config);
  const switching = useStore((s) => s.switching);
  const spots = map?.appleSpots ?? [];

  return (
    <section className="card world-menu" aria-labelledby="world-title">
      <header className="card-head">
        <div>
          <h2 id="world-title">World</h2>
          <p className="muted small">
            {map?.key === "open" ? "A house, a road loop, stairs, a park, a lake and hills" : "An empty stage for testing bodies"}
          </p>
        </div>
        <button type="button" className="icon-btn" aria-label="Close" onClick={() => setState({ worldMenu: false })}>
          <X size={18} />
        </button>
      </header>

      {config && (
        <div className="field map-field">
          <span>Map</span>
          <div className="segmented" role="radiogroup" aria-label="Map">
            {config.maps.map((m) => (
              <button key={m.key} type="button" role="radio" aria-checked={config.map === m.key} disabled={!!switching}
                className={config.map === m.key ? "is-on" : ""}
                onClick={() => config.map !== m.key && switchScene(m.key, config.bodies, `Opening the ${m.label.toLowerCase()}…`)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {spots.length > 1 && (
        <label className="field">
          <span>Apple</span>
          <select value={status?.apple.spot ?? ""} onChange={(e) => api.placeApple(e.target.value).catch(reportError)}>
            {spots.map((s) => <option key={s} value={s}>{sentence(s)}</option>)}
          </select>
        </label>
      )}

      <div className="field">
        <span>Speed</span>
        <div className="segmented" role="radiogroup" aria-label="Simulation speed">
          {SPEEDS.map((v) => (
            <button key={v} type="button" role="radio" aria-checked={status?.speed === v}
              className={status?.speed === v ? "is-on" : ""} onClick={() => api.speed(v).catch(reportError)}>
              {v}×
            </button>
          ))}
        </div>
      </div>

      <label className="toggle">
        <input type="checkbox" checked={topView} onChange={(e) => setState({ topView: e.target.checked })} />
        <span>Top view</span>
      </label>
      <label className="toggle">
        <input type="checkbox" checked={showCollision} onChange={(e) => setState({ showCollision: e.target.checked })} />
        <span>Show collision shapes</span>
      </label>

      <p className="muted small meta">
        Physics {!status ? "–" : status.paused ? "paused" : `${status.rtf.toFixed(2)}× real time`} · {fps || "–"} fps
      </p>
    </section>
  );
}
