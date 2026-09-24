import { lazy, Suspense, useEffect, useState } from "react";
import { Check, ChevronRight, Lock } from "lucide-react";
import { api, connectBrain, loadBrainPoints, reportError } from "../lib/api";
import { brainActivity, setState, useStore } from "../lib/store";

const BrainView = lazy(() => import("../brain/BrainView"));

const nf = new Intl.NumberFormat("en");

function pct(x: number | undefined): string {
  if (x === undefined) return "–";
  const v = x * 100;
  return v < 0.1 && v > 0 ? "<0.1%" : `${v < 10 ? v.toFixed(1) : Math.round(v)}%`;
}

/** The Brain tab of the dock: live 3D neurons, group activity, and the training roadmap. */
export default function BrainPanel() {
  const brain = useStore((s) => s.brain);
  const points = useStore((s) => s.brainPoints);
  const highlight = useStore((s) => s.brainHighlight);
  const [pointsError, setPointsError] = useState<string | null>(null);

  useEffect(() => {
    if (!brain?.available) return;
    loadBrainPoints().catch((e: Error) => setPointsError(e.message));
    const stop = connectBrain();
    return () => {
      stop();
      brainActivity.data = null;
      setState({ brainHighlight: null });
    };
  }, [brain?.available]);

  if (!brain?.available) return <p className="pad-text">{brain?.message ?? "Loading…"}</p>;
  const summary = brain.summary ?? brain.groups ?? [];

  return (
    <div className="brain-tab">
      <p className="muted">Real fruit-fly wiring (MaleCNS) · not trained yet</p>
      <label className="switch-row">
        <span>
          <strong>Listening to the bodies</strong>
          <span className="muted small">Their joint movements feed the brain's sense neurons.</span>
        </span>
        <input
          type="checkbox"
          role="switch"
          checked={brain.listening}
          onChange={(e) => api.listen(e.target.checked).then((r) =>
            setState({ brain: { ...brain, listening: r.listening } })).catch(reportError)}
        />
      </label>

      <div className="brain-view" aria-label="3D view of the fly nervous system. Drag to rotate.">
        {pointsError ? (
          <p className="muted center">Couldn't load the neuron positions: {pointsError}</p>
        ) : points ? (
          <Suspense fallback={<p className="muted center">Drawing 166k neurons…</p>}>
            <BrainView />
          </Suspense>
        ) : (
          <p className="muted center">Loading neuron positions…</p>
        )}
        <div className="brain-legend">
          <span><i className="dot rest" /> resting</span>
          <span><i className="dot hot" /> active</span>
          <span className="muted">brain on top, nerve cord below · drag to rotate</span>
        </div>
      </div>

      <div className="stats">
        <div><strong>{nf.format(brain.neurons ?? 0)}</strong><span>neurons</span></div>
        <div><strong>{((brain.connections ?? 0) / 1e6).toFixed(1)}M</strong><span>connections</span></div>
        <div><strong>{pct(brain.activeFraction)}</strong><span>active now</span></div>
      </div>

      <h3 className="section-title">Where the signals are</h3>
      <ul className="groups" onMouseLeave={() => setState({ brainHighlight: null })}>
        {summary.map((g, i) => (
          <li key={g.key}>
            <button
              type="button"
              className={`group-row ${highlight === i ? "is-on" : ""}`}
              onMouseEnter={() => setState({ brainHighlight: i })}
              onFocus={() => setState({ brainHighlight: i })}
              onClick={() => setState({ brainHighlight: highlight === i ? null : i })}
              aria-pressed={highlight === i}
            >
              <span className="group-name">
                <strong>{g.name}</strong>
                <span className="muted small">{g.description}</span>
              </span>
              <span className="group-level">
                <span className="bar" aria-hidden="true">
                  <span style={{ transform: `scaleX(${Math.min(1, (g.active ?? 0) * 4)})` }} />
                </span>
                <span className="small">{pct(g.active)}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      <p className="note">
        When a body moves, its joint sensors excite the <strong>Senses</strong> neurons and the signal
        spreads through the real fly wiring. Nothing is trained yet, so the brain only listens: it doesn't
        move the bodies. Hover a group to see where it is.
      </p>

      <h3 className="section-title">Training roadmap</h3>
      <ol className="steps">
        {(brain.stages ?? []).map((s) => (
          <li key={s.id} className={`step ${s.status}`}>
            <span className="step-icon" aria-hidden="true">
              {s.status === "done" ? <Check size={14} /> : s.status === "next" ? <ChevronRight size={14} /> : <Lock size={12} />}
            </span>
            <span>
              <strong>{s.name}</strong>
              <span className="muted small">{s.status === "done" ? "Done · " : s.status === "next" ? "Next · " : ""}{s.detail}</span>
            </span>
          </li>
        ))}
      </ol>
      <p className="muted small">
        {brain.kaggle
          ? "Kaggle is connected. Training will run on its free GPUs once stage 1 is built."
          : "Kaggle isn't connected yet. Training will run on its free GPUs once stage 1 is built."}
      </p>
      <p className="credit">Connectome: MaleCNS v1.0, Janelia FlyEM (CC-BY 4.0).</p>
    </div>
  );
}
