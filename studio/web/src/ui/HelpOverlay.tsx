import { X } from "lucide-react";
import { setState, useStore } from "../lib/store";

const GROUPS: { title: string; rows: [string, string][] }[] = [
  {
    title: "Look around",
    rows: [
      ["Drag", "Orbit the camera"],
      ["Right-drag", "Slide across the ground"],
      ["Scroll", "Zoom"],
      ["W A S D  or  arrows", "Move"],
      ["Q / E", "Turn left / right"],
      ["Z / X", "Zoom in / out"],
      ["T", "Top view on / off"],
      ["R", "Back to the start view"],
      ["Map (bottom left)", "Click anywhere to go there"],
    ],
  },
  {
    title: "Bodies",
    rows: [
      ["Click a body", "Inspect it"],
      ["Double-click a body", "Fly the camera to it"],
      ["1 – 9", "Inspect body 1, 2, 3…"],
      ["F", "Find the selected body"],
      ["Esc", "Close whatever is open"],
    ],
  },
  {
    title: "Drive and fly",
    rows: [
      ["Drive it / Fly it", "Pick this in the car's or drone's panel first"],
      ["Arrows", "Drive the car, or move the drone"],
      ["Shift + ↑ / ↓", "Drone height"],
    ],
  },
  {
    title: "Everything else",
    rows: [
      ["/", "Type a command"],
      ["Space", "Pause / resume"],
      ["B", "Brain"],
      ["?", "This help"],
    ],
  },
];

export default function HelpOverlay() {
  const open = useStore((s) => s.help);
  if (!open) return null;
  return (
    <div className="overlay" onClick={() => setState({ help: false })}>
      <section className="card help-card" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={(e) => e.stopPropagation()}>
        <header className="card-head">
          <div>
            <h2 id="help-title">Controls</h2>
            <p className="muted small">Everything also has a button on screen; the keys are shortcuts.</p>
          </div>
          <button type="button" className="icon-btn" aria-label="Close help" autoFocus onClick={() => setState({ help: false })}>
            <X size={18} />
          </button>
        </header>
        <div className="help-grid">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <h3 className="section-title">{g.title}</h3>
              <dl className="keys">
                {g.rows.map(([k, v]) => (
                  <div key={k}>
                    <dt><kbd>{k}</kbd></dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
