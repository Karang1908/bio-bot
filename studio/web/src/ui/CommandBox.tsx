import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { ArrowUp, CircleAlert, MessageSquare, X } from "lucide-react";
import { api } from "../lib/api";
import { setState, useStore } from "../lib/store";

/** Example commands that make sense for this map and the bodies in it. */
function suggestions(map: string | undefined, bodies: string[]): string[] {
  const has = (k: string) => bodies.includes(k);
  const out: string[] = [];
  if (map === "open") {
    if (has("car")) out.push("Drive to the lake");
    if (has("drone")) out.push("Drone, fly to the hills");
    if (has("dog")) out.push("Dog, swim");
    out.push("Find the apple", "Put the apple on the platform");
  } else {
    out.push("Find the apple");
    if (has("drone")) out.push("Drone, go to the apple");
  }
  out.push("Everyone explore", "Everyone stand");
  return out.slice(0, 5);
}

export interface CommandHandle { focus: () => void }

const CommandBox = forwardRef<CommandHandle>(function CommandBox(_, ref) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const history = useRef<string[]>([]);
  const hIndex = useRef(-1);
  const reply = useStore((s) => s.reply);
  const live = useStore((s) => s.link === "open");
  const map = useStore((s) => s.scene?.map?.key);
  const lineup = useStore((s) => s.scene?.lineup);

  useImperativeHandle(ref, () => ({ focus: () => input.current?.focus() }), []);

  const send = async (value: string) => {
    const t = value.trim();
    if (!t || busy) return;
    setBusy(true);
    history.current = [t, ...history.current.filter((h) => h !== t)].slice(0, 30);
    hIndex.current = -1;
    try {
      const r = await api.prompt(t);
      setState({ reply: { id: Date.now(), text: t, reply: r.reply, error: false } });
      setText("");
    } catch (e) {
      setState({ reply: { id: Date.now(), text: t, reply: (e as Error).message, error: true } });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="command">
      {reply ? (
        <div className={`reply ${reply.error ? "error" : ""}`} role="status" aria-live="polite">
          {reply.error ? <CircleAlert size={16} aria-hidden="true" /> : <MessageSquare size={16} aria-hidden="true" />}
          <p>
            {reply.text && <span className="said">“{reply.text}”</span>}
            {reply.reply}
          </p>
          <button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => setState({ reply: null })}>
            <X size={16} />
          </button>
        </div>
      ) : (
        <div className="suggestions" aria-label="Examples">
          {suggestions(map, lineup ?? []).map((s) => (
            <button key={s} type="button" className="suggestion" disabled={!live} onClick={() => void send(s)}>{s}</button>
          ))}
        </div>
      )}
      <form className="command-box" onSubmit={(e) => { e.preventDefault(); void send(text); }}>
        <input
          ref={input}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
            const h = history.current;
            if (!h.length) return;
            e.preventDefault();
            hIndex.current = Math.max(-1, Math.min(h.length - 1, hIndex.current + (e.key === "ArrowUp" ? 1 : -1)));
            setText(hIndex.current < 0 ? "" : h[hIndex.current]);
          }}
          placeholder={live ? "Tell the bodies what to do…" : "Waiting for the studio server…"}
          aria-label="Command"
          disabled={!live}
          autoComplete="off"
          spellCheck={false}
        />
        <button type="submit" className="send" aria-label="Send" disabled={!text.trim() || busy || !live}>
          <ArrowUp size={18} />
        </button>
      </form>
    </div>
  );
});

export default CommandBox;
