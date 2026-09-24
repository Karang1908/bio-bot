import type { SceneDef } from "../lib/types";

/** Where the camera starts for each map. Kept free of three.js so the UI can import it cheaply. */
export function homeView(scene: SceneDef): { x: number; y: number; distance: number } {
  if (scene.map?.key === "open") return { x: 0, y: 0, distance: 16 };
  return { x: 0, y: 0, distance: Math.max(6.5, 3.4 * scene.lineup.length) };
}
