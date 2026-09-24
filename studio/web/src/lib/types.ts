export type Mode = "off" | "hold" | "babble" | "manual";

export interface GeomDef {
  id: number;
  name: string;
  body: number;
  agent: string | null;
  type: "plane" | "hfield" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box" | "mesh" | "sdf";
  size: [number, number, number];
  pos: [number, number, number];
  quat: [number, number, number, number]; // MuJoCo order: w x y z
  rgba: [number, number, number, number];
  metallic: number;
  roughness: number;
  emission: number;
  mesh: number | null;
  hfield: number | null;
  visual: boolean;
}

export interface HFieldDef {
  id: number;
  nrow: number;
  ncol: number;
  halfX: number;
  halfY: number;
  offset: number; // byte offset of float32 heights (metres) in the mesh blob
  geom: number | null;
}

export interface PlaceDef { key: string; label: string; x: number; y: number; view: number }
export interface WaterDef { cx: number; cy: number; rx: number; ry: number; level: number }
export interface MapDef {
  key: "sandbox" | "open";
  label: string;
  size: number;
  places: PlaceDef[];
  water: WaterDef[];
  appleSpots: string[];
}

export interface MeshDef {
  id: number;
  name: string;
  vertOffset: number;
  vertCount: number;
  faceOffset: number;
  faceCount: number;
}

export interface AgentDef {
  key: string;
  label: string;
  credit: string;
  notes: string;
  kind: "legged" | "aerial" | "vehicle";
  root: number;
  actuators: number;
  sensors: number;
  joints: number;
  mass: number;
  magnification: number | null;
}

export interface SceneDef {
  bodies: { id: number; name: string; agent: string | null }[];
  geoms: GeomDef[];
  meshes: MeshDef[];
  hfields: HFieldDef[];
  meshBytes: number;
  agents: AgentDef[];
  objects: Record<string, number>;
  timestep: number;
  map: MapDef | null;
  lineup: string[];
  version: number;
}

export interface ConfigInfo {
  map: "sandbox" | "open";
  bodies: string[];
  version: number;
  maps: { key: "sandbox" | "open"; label: string }[];
  available: string[];
  all: { key: string; label: string; kind: string }[];
}

export interface InspectInfo {
  key: string;
  label: string;
  kind: string;
  mass: number;
  dof: number;
  parts: { id: number; name: string }[];
  joints: { name: string; type: string; range: [number, number] | null; part: number; qadr: number }[];
  actuators: { index: number; name: string; kind: string; joint: string | null; part: number; range: [number, number]; unit: string }[];
  sensors: { name: string; type: string; dim: number; adr: number }[];
}

export interface LiveInfo {
  key: string;
  version: number;
  mode: Mode;
  joints: number[];
  ctrl: number[];
  force: number[];
  manual: number[];
  sensors: number[];
  drive?: { throttle: number; steer: number; speed: number };
  setpoint?: [number, number, number];
}

export interface BodyStatus {
  mode: Mode;
  connected: boolean;
  strength: number;
  position: [number, number, number];
  upright: number;
  speed: number;
  setpoint: [number, number, number] | null;
  target: [number, number] | null;
  wet: boolean;
}

export interface Goal {
  text: string;
  verb: string | null;
  target: string | null;
  bodies: string[];
  status: string;
  executed_by: string | null;
  point: [number, number, number] | null;
}

export interface LogEntry {
  t: number;
  kind: string;
  text: string;
}

export interface Status {
  time: number;
  rtf: number;
  speed: number;
  paused: boolean;
  bodies: Record<string, BodyStatus>;
  apple: { spot: string; position: [number, number, number] };
  goal: Goal | null;
  log: LogEntry[];
  version: number;
}

export interface BrainGroup {
  key: string;
  name: string;
  description: string;
  neurons?: number;
  mean?: number;
  active?: number;
}

export interface BrainStage {
  id: string;
  name: string;
  detail: string;
  status: "done" | "next" | "locked";
}

export interface BrainInfo {
  available: boolean;
  listening: boolean;
  groups: BrainGroup[];
  message?: string;
  neurons?: number;
  connections?: number;
  synapses?: number;
  drawn?: number;
  steps?: number;
  stepMs?: number;
  activeFraction?: number;
  summary?: BrainGroup[];
  built?: boolean;
  trained?: boolean;
  stages?: BrainStage[];
  kaggle?: boolean;
}

export interface PromptReply {
  parsed: { verb: string | null; bodies: string[]; target: string | null; spot: string | null };
  actions: string[];
  reply: string;
}
