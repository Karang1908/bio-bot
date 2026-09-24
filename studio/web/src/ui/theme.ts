import { Bug, CarFront, Dog, Drone, PersonStanding, type LucideIcon } from "lucide-react";
import type { AgentDef, BodyStatus, Mode } from "../lib/types";

export interface BodyLook {
  color: string;
  icon: LucideIcon;
  model: string;
}

export const BODY: Record<string, BodyLook> = {
  humanoid: { color: "#F59E0B", icon: PersonStanding, model: "Unitree G1" },
  dog: { color: "#2DD4BF", icon: Dog, model: "Unitree Go2" },
  drone: { color: "#60A5FA", icon: Drone, model: "Skydio X2" },
  fly: { color: "#C084FC", icon: Bug, model: "Fruit fly (flybody), 100× size" },
  car: { color: "#F472B6", icon: CarFront, model: "Go-kart" },
};

export function look(key: string): BodyLook {
  return BODY[key] ?? { color: "#E4E4E7", icon: PersonStanding, model: key };
}

type Kind = AgentDef["kind"] | string;

/** What each built-in mode is called for a body, in plain words. */
export function modeLabel(kind: Kind, mode: Mode): string {
  if (mode === "off") return "Relax";
  if (mode === "babble") return "Explore";
  if (mode === "manual") return kind === "vehicle" ? "Drive it" : kind === "aerial" ? "Fly it" : "Take control";
  return kind === "aerial" ? "Hover" : kind === "vehicle" ? "Brake" : "Stand";
}

export function modeHelp(kind: Kind, mode: Mode): string {
  if (mode === "off") return "Power off. The body goes limp.";
  if (mode === "babble") return "Random movements: the first step of learning its own body.";
  if (mode === "manual") {
    if (kind === "vehicle") return "You drive: arrow keys or the pad below.";
    if (kind === "aerial") return "You fly: arrow keys move it, the Up/Down buttons change height.";
    return "You pose it: drag any joint's slider below.";
  }
  if (kind === "aerial") return "Hover in place with the built-in flight controller.";
  if (kind === "vehicle") return "Stay parked (or drive to a place you pick below).";
  return "Hold still with the built-in balance controller.";
}

/** One-word condition shown on chips and in the inspector. */
export function condition(a: AgentDef, b: BodyStatus | undefined): { word: string; bad: boolean } {
  if (!b) return { word: "…", bad: false };
  if (b.mode === "off") return { word: b.wet ? "Floating" : "Relaxed", bad: false };
  if (a.kind === "vehicle") {
    if (b.upright < 0.3) return { word: "Flipped", bad: true };
    if (b.target) return { word: "Driving", bad: false };
    return { word: b.speed > 0.3 ? "Moving" : "Parked", bad: false };
  }
  if (a.kind === "aerial") {
    if (b.upright < 0.3) return { word: "Crashed", bad: true };
    return { word: b.position[2] > 0.3 ? (b.mode === "babble" ? "Exploring" : "Flying") : "Landed", bad: false };
  }
  if (b.wet) return { word: "Swimming", bad: false };
  if (b.upright < 0.5) return { word: "Fallen", bad: true };
  return { word: b.mode === "babble" ? "Exploring" : b.mode === "manual" ? "Posed" : "Standing", bad: false };
}

/** "left_knee_joint" -> "Left knee". */
export function pretty(name: string): string {
  const s = name.replace(/_(joint|link|collision)$/g, "").replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}
