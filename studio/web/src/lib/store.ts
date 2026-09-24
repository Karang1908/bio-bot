import { useSyncExternalStore } from "react";
import type { BrainInfo, ConfigInfo, InspectInfo, LiveInfo, SceneDef, Status } from "./types";

export type Link = "connecting" | "open" | "closed";
export type DockTab = "body" | "brain";

export interface Reply {
  id: number;
  text: string;
  reply: string;
  error: boolean;
}

export interface BrainPoints {
  count: number;
  positions: Float32Array; // x y z in mm, centred
  groups: Uint8Array;
}

export interface State {
  link: Link;
  status: Status | null;
  scene: SceneDef | null;
  meshes: ArrayBuffer | null;
  loadProgress: number; // 0..1 while loading meshes
  loadError: string | null;
  switching: string | null; // message while the server rebuilds the world
  config: ConfigInfo | null;
  selected: string | null; // body shown in the inspector
  inspect: InspectInfo | null;
  live: LiveInfo | null;
  hoverPart: number | null; // MuJoCo body id to highlight in 3D
  dock: DockTab | null; // right-hand dock: body inspector or brain
  follow: boolean;
  topView: boolean;
  showCollision: boolean;
  help: boolean;
  worldMenu: boolean;
  brain: BrainInfo | null;
  brainPoints: BrainPoints | null;
  brainHighlight: number | null; // group index highlighted in the 3D brain
  reply: Reply | null;
  fps: number;
}

let state: State = {
  link: "connecting",
  status: null,
  scene: null,
  meshes: null,
  loadProgress: 0,
  loadError: null,
  switching: null,
  config: null,
  selected: null,
  inspect: null,
  live: null,
  hoverPart: null,
  dock: null,
  follow: false,
  topView: false,
  showCollision: false,
  help: false,
  worldMenu: false,
  brain: null,
  brainPoints: null,
  brainHighlight: null,
  reply: null,
  fps: 0,
};

const listeners = new Set<() => void>();

export function getState(): State {
  return state;
}

export function setState(patch: Partial<State>): void {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** Select a slice. Selectors must return primitives or references already held in state. */
export function useStore<T>(select: (s: State) => T): T {
  return useSyncExternalStore(subscribe, () => select(state));
}

/**
 * High-rate data lives outside React. The world WebSocket writes ~30 pose frames/s
 * here; the brain WebSocket writes ~10 activity frames/s. The 3D views read them
 * every animation frame.
 */
export const poses = {
  prev: null as Float32Array | null,
  curr: null as Float32Array | null,
  prevAt: 0,
  currAt: 0,
};

export const brainActivity = {
  data: null as Uint8Array | null,
  frame: 0,
};

/** Camera state shared with the minimap, and requests for the camera to move. */
export const camera = {
  x: 0, // orbit target on the ground (MuJoCo coordinates)
  y: 0,
  yaw: 0, // direction the camera looks, radians (MuJoCo x-y plane)
  distance: 10,
  request: null as null | { x: number; y: number; z?: number; distance?: number },
  nav: { rotate: 0, zoom: 0, reset: false },
};

export function flyTo(x: number, y: number, distance?: number, z?: number): void {
  camera.request = { x, y, z, distance };
}
