import { lazy, Suspense, useEffect, useRef } from "react";
import { CircleAlert } from "lucide-react";
import { api, connectWorld, loadWorld, reportError } from "./lib/api";
import { getState, setState, useStore } from "./lib/store";
import CommandBox, { type CommandHandle } from "./ui/CommandBox";
import Dock from "./ui/Dock";
import HelpOverlay from "./ui/HelpOverlay";
import Navigator from "./ui/Navigator";
import TopBar from "./ui/TopBar";
import WorldMenu from "./ui/WorldMenu";

const World = lazy(() => import("./world/World"));

function WorldStage() {
  const meshes = useStore((s) => s.meshes);
  const progress = useStore((s) => s.loadProgress);
  const error = useStore((s) => s.loadError);
  const scene = useStore((s) => s.scene);
  const link = useStore((s) => s.link);

  if (error || (link === "closed" && !scene)) {
    return (
      <div className="stage-msg">
        <div className="card msg">
          <CircleAlert size={20} className="bad" aria-hidden="true" />
          <div>
            <h2>Can't reach the studio server</h2>
            <p className="muted">Start it from the project folder with <code>scripts/studio.sh</code>. This page reconnects by itself.</p>
            {error && <p className="muted small">Details: {error}</p>}
          </div>
        </div>
      </div>
    );
  }
  if (!meshes) {
    const total = scene ? scene.meshBytes / 1e6 : 0;
    return (
      <div className="stage-msg">
        <div className="card msg">
          <div className="grow">
            <h2>Loading the world…</h2>
            <p className="muted">Body models and terrain ({(progress * total).toFixed(0)} of {total ? total.toFixed(0) : "…"} MB)</p>
            <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
              <span style={{ transform: `scaleX(${progress})` }} />
            </div>
          </div>
        </div>
      </div>
    );
  }
  return (
    <Suspense fallback={<div className="stage-msg"><div className="card msg"><h2>Building the scene…</h2></div></div>}>
      <World />
    </Suspense>
  );
}

/** Shown while the server rebuilds the world for a new map or lineup. */
function Switching() {
  const message = useStore((s) => s.switching);
  if (!message) return null;
  return (
    <div className="switching" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {message}
    </div>
  );
}

export default function App() {
  const cmd = useRef<CommandHandle>(null);
  const dock = useStore((s) => s.dock);
  const worldMenu = useStore((s) => s.worldMenu);

  useEffect(() => {
    void loadWorld();
    return connectWorld();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = getState();
      if (e.key === "Escape") {
        if (s.help) setState({ help: false });
        else if (s.worldMenu) setState({ worldMenu: false });
        else if (s.dock) setState({ dock: null, selected: null, follow: false });
        return;
      }
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      const agents = s.scene?.agents ?? [];
      if (e.key === "/") {
        e.preventDefault();
        cmd.current?.focus();
      } else if (e.key === "?") {
        setState({ help: !s.help, worldMenu: false });
      } else if (/^[1-9]$/.test(e.key) && agents[Number(e.key) - 1]) {
        const k = agents[Number(e.key) - 1].key;
        setState(s.selected === k && s.dock === "body" ? { selected: null, dock: null } : { selected: k, dock: "body" });
      } else if (e.key === "b") {
        setState({ dock: s.dock === "brain" ? (s.selected ? "body" : null) : "brain" });
      } else if (e.key === " " && s.status && (e.target === document.body || t?.tagName === "CANVAS")) {
        e.preventDefault();
        api.pause(!s.status.paused).catch(reportError);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className={`app ${dock ? "dock-open" : ""}`}>
      <main className="stage" aria-label="3D world">
        <WorldStage />
      </main>
      <TopBar />
      {worldMenu && <WorldMenu />}
      <Navigator />
      <CommandBox ref={cmd} />
      <Dock />
      <Switching />
      <HelpOverlay />
    </div>
  );
}
