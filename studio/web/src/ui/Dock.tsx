import { lazy, Suspense } from "react";
import { Brain, SquareUser, X } from "lucide-react";
import { getState, setState, useStore } from "../lib/store";
import Inspector from "./Inspector";

const BrainPanel = lazy(() => import("./BrainPanel"));

/** Right-hand dock with two tabs: the selected body's inspector, and the brain. */
export default function Dock() {
  const dock = useStore((s) => s.dock);
  const selected = useStore((s) => s.selected);
  if (!dock) return null;
  return (
    <aside className="dock" aria-label={dock === "brain" ? "Brain" : "Body inspector"}>
      <div className="dock-tabs" role="tablist">
        <button type="button" role="tab" aria-selected={dock === "body"} className={dock === "body" ? "is-on" : ""}
          onClick={() => setState({ dock: "body" })}>
          <SquareUser size={16} /> {selected ? "Body" : "Bodies"}
        </button>
        <button type="button" role="tab" aria-selected={dock === "brain"} className={dock === "brain" ? "is-on" : ""}
          onClick={() => setState({ dock: "brain" })}>
          <Brain size={16} /> Brain
        </button>
        <button type="button" className="icon-btn" aria-label="Close panel"
          onClick={() => setState({ dock: null, selected: getState().dock === "body" ? null : getState().selected })}>
          <X size={18} />
        </button>
      </div>
      <div className="dock-body">
        {dock === "body" ? <Inspector /> : (
          <Suspense fallback={<p className="pad-text muted">Loading…</p>}><BrainPanel /></Suspense>
        )}
      </div>
    </aside>
  );
}
