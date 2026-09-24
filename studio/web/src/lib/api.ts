import { brainActivity, getState, poses, setState } from "./store";
import type { BrainInfo, ConfigInfo, InspectInfo, LiveInfo, PromptReply, SceneDef, Status } from "./types";

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* body was not JSON; keep the status line */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  setMode: (key: string, mode: string) => request("POST", `/api/bodies/${key}/mode`, { mode }),
  setStrength: (key: string, value: number) => request("POST", `/api/bodies/${key}/strength`, { value }),
  resetBody: (key: string) => request("POST", `/api/bodies/${key}/reset`),
  manual: (key: string, index: number, value: number) => request("POST", `/api/bodies/${key}/manual`, { index, value }),
  drive: (key: string, throttle: number, steer: number) => request("POST", `/api/bodies/${key}/drive`, { throttle, steer }),
  nudge: (key: string, dx: number, dy: number, dz: number) => request("POST", `/api/bodies/${key}/nudge`, { dx, dy, dz }),
  goto: (key: string, place: string) => request("POST", `/api/bodies/${key}/goto`, { place }),
  inspect: (key: string) => request<InspectInfo>("GET", `/api/bodies/${key}/inspect`),
  resetWorld: () => request("POST", "/api/world/reset"),
  pause: (paused: boolean) => request("POST", "/api/world/pause", { paused }),
  speed: (value: number) => request("POST", "/api/world/speed", { value }),
  placeApple: (spot: string) => request("POST", "/api/world/apple", { spot }),
  prompt: (text: string) => request<PromptReply>("POST", "/api/prompt", { text }),
  listen: (on: boolean) => request<{ listening: boolean }>("POST", "/api/brain/listen", { on }),
  brain: () => request<BrainInfo>("GET", "/api/brain"),
  config: () => request<ConfigInfo>("GET", "/api/config"),
};

/** Report a failed action in the reply area instead of failing silently. */
export function reportError(e: unknown): void {
  setState({ reply: { id: Date.now(), text: "", reply: e instanceof Error ? e.message : String(e), error: true } });
}

let loading: Promise<void> | null = null;

/** Load the scene, the mesh blob (with progress), config, and brain info. */
export function loadWorld(force = false): Promise<void> {
  if (force) loading = null;
  loading ??= (async () => {
    try {
      const [scene, config, brain] = await Promise.all([
        request<SceneDef>("GET", "/api/scene"),
        request<ConfigInfo>("GET", "/api/config"),
        getState().brain ? Promise.resolve(getState().brain!) : request<BrainInfo>("GET", "/api/brain"),
      ]);
      setState({ config, brain, loadError: null, loadProgress: 0 });
      const res = await fetch("/api/meshes.bin");
      if (!res.ok || !res.body) throw new Error(`meshes: ${res.status}`);
      const reader = res.body.getReader();
      const out = new Uint8Array(scene.meshBytes);
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (got + value.length > out.length) throw new Error("meshes: more data than expected");
        out.set(value, got);
        got += value.length;
        setState({ loadProgress: got / Math.max(scene.meshBytes, 1) });
      }
      if (got !== scene.meshBytes) throw new Error(`meshes truncated: ${got} of ${scene.meshBytes} bytes`);
      const selected = getState().selected;
      setState({
        scene, meshes: out.buffer, loadProgress: 1, switching: null,
        selected: selected && scene.lineup.includes(selected) ? selected : null,
      });
    } catch (e) {
      loading = null;
      setState({ loadError: e instanceof Error ? e.message : String(e), switching: null });
    }
  })();
  return loading;
}

/** Ask the server for a different map or lineup, then reload the scene. */
export async function switchScene(map: "sandbox" | "open", bodies: string[], message: string): Promise<void> {
  setState({ switching: message });
  try {
    await request("POST", "/api/config", { map, bodies });
    await loadWorld(true);
  } catch (e) {
    setState({ switching: null });
    reportError(e);
  }
}

/** Brain neuron positions + groups (once). */
export async function loadBrainPoints(): Promise<void> {
  if (getState().brainPoints) return;
  const res = await fetch("/api/brain/points.bin");
  if (!res.ok) throw new Error(`brain points: ${res.status}`);
  const buf = await res.arrayBuffer();
  const count = new Uint32Array(buf, 0, 1)[0];
  const positions = new Float32Array(buf, 4, count * 3);
  const groups = new Uint8Array(buf, 4 + count * 12, count);
  setState({ brainPoints: { count, positions, groups } });
}

function socket(path: string, onBinary: (b: ArrayBuffer) => void, onText: (t: string) => void,
  onLink?: (open: boolean | null) => void): () => void {
  let ws: WebSocket | null = null;
  let retry = 250;
  let closed = false;
  let timer = 0;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}${path}`);
    ws.binaryType = "arraybuffer";
    onLink?.(null);
    ws.onopen = () => {
      retry = 250;
      onLink?.(true);
    };
    ws.onmessage = (ev) => (typeof ev.data === "string" ? onText(ev.data) : onBinary(ev.data as ArrayBuffer));
    ws.onclose = () => {
      if (closed) return;
      onLink?.(false);
      timer = window.setTimeout(open, retry);
      retry = Math.min(retry * 2, 3000);
    };
  };
  open();
  return () => {
    closed = true;
    window.clearTimeout(timer);
    ws?.close();
  };
}

/** World stream: body poses (binary) and status (JSON). Reloads the scene if the server rebuilt it. */
export function connectWorld(): () => void {
  return socket(
    "/ws",
    (buf) => {
      poses.prev = poses.curr;
      poses.prevAt = poses.currAt;
      poses.curr = new Float32Array(buf, 12);
      poses.currAt = performance.now();
    },
    (text) => {
      const msg = JSON.parse(text) as { type: string } & Status;
      if (msg.type !== "status") return;
      setState({ status: msg });
      const s = getState();
      if (s.scene && msg.version !== s.scene.version && !s.switching) {
        setState({ switching: "Loading the new scene…" });
        void loadWorld(true);
      }
    },
    (open) => {
      setState({ link: open === null ? "connecting" : open ? "open" : "closed" });
      if (open === false && (getState().loadError || !getState().meshes)) void loadWorld(true);
    },
  );
}

/** Live joints, actuators and sensors of one body (while the inspector shows it). */
export function connectInspect(key: string): () => void {
  return socket(
    `/ws/inspect/${key}`,
    () => undefined,
    (text) => {
      const msg = JSON.parse(text) as LiveInfo & { type: string };
      if (msg.type === "live" && msg.key === getState().selected) setState({ live: msg });
    },
  );
}

/** Brain stream (only while the brain tab is open): activity bytes and a JSON summary. */
export function connectBrain(): () => void {
  return socket(
    "/ws/brain",
    (buf) => {
      brainActivity.data = new Uint8Array(buf);
      brainActivity.frame++;
    },
    (text) => {
      const msg = JSON.parse(text) as { type: string } & BrainInfo;
      if (msg.type === "brain") setState({ brain: { ...getState().brain, ...msg } });
    },
  );
}
