import * as THREE from "three";

/** Small procedural textures, drawn once on a canvas (no external image assets). */

function canvasTexture(size: number, draw: (c: CanvasRenderingContext2D, s: number) => void, repeat: [number, number]) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  draw(ctx, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat[0], repeat[1]);
  tex.anisotropy = 8;
  return tex;
}

function rand(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/** Oak planks, 8 m x 6 m room: one tile = 2 m x 2 m. */
export function floorTexture(): THREE.Texture {
  return canvasTexture(1024, (c, s) => {
    const r = rand(7);
    const plankH = s / 10;
    for (let row = 0; row < 10; row++) {
      let x = -r() * s * 0.5;
      while (x < s) {
        const len = s * (0.35 + r() * 0.4);
        const tone = 118 + r() * 30;
        c.fillStyle = `rgb(${tone + 40}, ${tone + 8}, ${tone - 30})`;
        c.fillRect(x, row * plankH, len, plankH);
        for (let i = 0; i < 26; i++) { // grain
          c.strokeStyle = `rgba(70, 40, 20, ${0.05 + r() * 0.08})`;
          c.lineWidth = 1 + r() * 1.5;
          const y = row * plankH + r() * plankH;
          c.beginPath();
          c.moveTo(x, y);
          c.bezierCurveTo(x + len * 0.3, y + (r() - 0.5) * 6, x + len * 0.7, y + (r() - 0.5) * 6, x + len, y);
          c.stroke();
        }
        c.fillStyle = "rgba(40, 24, 12, 0.55)";
        c.fillRect(x, row * plankH, 2, plankH);
        x += len;
      }
      c.fillStyle = "rgba(40, 24, 12, 0.5)";
      c.fillRect(0, row * plankH, s, 2);
    }
  }, [4, 3]);
}

/** Woven wool rug. */
export function rugTexture(): THREE.Texture {
  return canvasTexture(512, (c, s) => {
    const r = rand(3);
    c.fillStyle = "#8a4a3c";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 9000; i++) {
      const v = r();
      c.fillStyle = v > 0.5 ? `rgba(255, 220, 190, ${0.03 + r() * 0.05})` : `rgba(40, 10, 5, ${0.05 + r() * 0.08})`;
      c.fillRect(r() * s, r() * s, 2, 1 + r() * 3);
    }
    c.strokeStyle = "rgba(230, 200, 160, 0.35)";
    c.lineWidth = 6;
    c.strokeRect(24, 24, s - 48, s - 48);
    c.lineWidth = 2;
    c.strokeRect(40, 40, s - 80, s - 80);
  }, [1, 1]);
}

/** Asphalt with a dashed centre line; `along` is the road's long axis in texture space. */
export function roadTexture(along: "u" | "v", repeat: number): THREE.Texture {
  return canvasTexture(512, (c, s) => {
    const r = rand(21);
    c.fillStyle = "#2c2d31";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 14000; i++) {
      c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.05})` : `rgba(0,0,0,${r() * 0.12})`;
      c.fillRect(r() * s, r() * s, 1.5, 1.5);
    }
    c.fillStyle = "rgba(235, 220, 170, 0.85)";
    for (let k = 0; k < 4; k++) {
      if (along === "v") c.fillRect(s / 2 - 5, k * (s / 4) + 16, 10, s / 8);
      else c.fillRect(k * (s / 4) + 16, s / 2 - 5, s / 8, 10);
    }
    c.fillStyle = "rgba(240, 240, 235, 0.7)"; // edge lines
    if (along === "v") { c.fillRect(10, 0, 6, s); c.fillRect(s - 16, 0, 6, s); }
    else { c.fillRect(0, 10, s, 6); c.fillRect(0, s - 16, s, 6); }
  }, along === "v" ? [1, repeat] : [repeat, 1]);
}

/** Studio floor for the sandbox: light concrete with 1 m and 5 m grid lines. One tile = 5 m. */
export function studioTexture(repeat: number): THREE.Texture {
  return canvasTexture(1024, (c, s) => {
    const r = rand(5);
    c.fillStyle = "#9a9d9f";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 20000; i++) {
      c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.035})` : `rgba(0,0,0,${r() * 0.05})`;
      c.fillRect(r() * s, r() * s, 2, 2);
    }
    c.fillStyle = "rgba(40, 44, 48, 0.28)";
    for (let k = 0; k <= 5; k++) {
      c.fillRect((k * s) / 5 - 1, 0, 2, s);
      c.fillRect(0, (k * s) / 5 - 1, s, 2);
    }
    c.fillStyle = "rgba(40, 44, 48, 0.5)";
    c.fillRect(0, 0, 4, s);
    c.fillRect(0, 0, s, 4);
  }, [repeat, repeat]);
}

/** Draw a mark and its wrapped copies, so the texture tiles without seams. */
function tiled(s: number, x: number, y: number, pad: number, draw: (x: number, y: number) => void) {
  for (const dx of x < pad ? [0, s] : x > s - pad ? [0, -s] : [0]) {
    for (const dy of y < pad ? [0, s] : y > s - pad ? [0, -s] : [0]) draw(x + dx, y + dy);
  }
}

/** Grass detail, kept light so the terrain's vertex colours set the hue. One tile = 4 m. */
export function grassTexture(repeat: number): THREE.Texture {
  return canvasTexture(512, (c, s) => {
    const r = rand(31);
    c.fillStyle = "#dde6cc";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 2600; i++) { // soft clumps
      const x = r() * s, y = r() * s, rad = 6 + r() * 18, light = r() > 0.5;
      tiled(s, x, y, rad, (px, py) => {
        const g = c.createRadialGradient(px, py, 0, px, py, rad);
        g.addColorStop(0, light ? "rgba(255,255,235,0.10)" : "rgba(70,90,40,0.10)");
        g.addColorStop(1, "rgba(0,0,0,0)");
        c.fillStyle = g;
        c.fillRect(px - rad, py - rad, 2 * rad, 2 * rad);
      });
    }
    for (let i = 0; i < 14000; i++) { // blades
      const x = r() * s, y = r() * s, len = 3 + r() * 7, a = -Math.PI / 2 + (r() - 0.5) * 1.1;
      const light = r() > 0.55;
      c.strokeStyle = light ? `rgba(255,255,225,${0.18 + r() * 0.22})` : `rgba(55,75,28,${0.10 + r() * 0.16})`;
      c.lineWidth = 0.8 + r() * 0.9;
      tiled(s, x, y, 10, (px, py) => {
        c.beginPath();
        c.moveTo(px, py);
        c.lineTo(px + Math.cos(a) * len, py + Math.sin(a) * len);
        c.stroke();
      });
    }
  }, [repeat, repeat]);
}

/** Concrete paving slabs, 0.5 m square. One tile = 2 m (use with world-scale UVs). */
export function pavingTexture(): THREE.Texture {
  return canvasTexture(512, (c, s) => {
    const r = rand(41);
    const n = 4, w = s / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const t = 178 + r() * 22;
        c.fillStyle = `rgb(${t}, ${t - 2}, ${t - 7})`;
        c.fillRect(i * w, j * w, w, w);
      }
    }
    for (let i = 0; i < 9000; i++) {
      c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.06})` : `rgba(0,0,0,${r() * 0.08})`;
      c.fillRect(r() * s, r() * s, 1.5, 1.5);
    }
    c.fillStyle = "rgba(60, 58, 54, 0.55)";
    for (let k = 0; k <= n; k++) {
      c.fillRect(k * w - 1.5, 0, 3, s);
      c.fillRect(0, k * w - 1.5, s, 3);
    }
  }, [1, 1]);
}

export type Facade = "brick" | "concrete" | "glass";

/** One 3 m x 3 m bay of a building facade: wall, window and floor line (use with world-scale UVs). */
export function facadeTexture(kind: Facade): THREE.Texture {
  return canvasTexture(256, (c, s) => {
    const r = rand(kind.length * 13);
    const glass = (x: number, y: number, w: number, h: number) => {
      const g = c.createLinearGradient(x, y, x + w * 0.4, y + h);
      g.addColorStop(0, "#9fb6c8");
      g.addColorStop(0.45, "#3d5264");
      g.addColorStop(1, "#22303c");
      c.fillStyle = g;
      c.fillRect(x, y, w, h);
    };
    if (kind === "brick") {
      c.fillStyle = "#8e4a36";
      c.fillRect(0, 0, s, s);
      const bh = s / 24, bw = s / 8;
      for (let row = 0; row < 24; row++) {
        for (let col = -1; col < 9; col++) {
          const t = r() * 26 - 13;
          c.fillStyle = `rgb(${150 + t}, ${78 + t * 0.6}, ${58 + t * 0.4})`;
          c.fillRect(col * bw + (row % 2) * bw * 0.5 + 1, row * bh + 1, bw - 2, bh - 2);
        }
      }
      c.fillStyle = "#e9e4da";
      c.fillRect(s * 0.24, s * 0.2, s * 0.52, s * 0.58);            // frame
      glass(s * 0.27, s * 0.23, s * 0.46, s * 0.52);
      c.fillStyle = "#e9e4da";
      c.fillRect(s * 0.495, s * 0.23, s * 0.01, s * 0.52);          // mullion
      c.fillStyle = "#cfc8bb";
      c.fillRect(s * 0.21, s * 0.78, s * 0.58, s * 0.035);          // sill
    } else if (kind === "concrete") {
      c.fillStyle = "#cbc6bb";
      c.fillRect(0, 0, s, s);
      for (let i = 0; i < 3000; i++) {
        c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.08})` : `rgba(0,0,0,${r() * 0.07})`;
        c.fillRect(r() * s, r() * s, 2, 2);
      }
      glass(0, s * 0.3, s, s * 0.44);                               // ribbon window
      c.fillStyle = "#b9b3a6";
      for (let k = 0; k < 4; k++) c.fillRect((k * s) / 4 - 2, s * 0.3, 4, s * 0.44);
      c.fillStyle = "rgba(0,0,0,0.18)";
      c.fillRect(0, s * 0.74, s, 3);
    } else {
      glass(0, 0, s, s);
      c.fillStyle = "rgba(255,255,255,0.06)";
      c.fillRect(0, 0, s * 0.5, s);
      c.fillStyle = "#56626d";                                        // spandrel at the slab
      c.fillRect(0, s * 0.86, s, s * 0.14);
      c.fillStyle = "#9aa6b0";                                        // mullions
      c.fillRect(0, 0, 4, s);
      c.fillRect(s / 2 - 2, 0, 4, s);
      c.fillRect(0, s * 0.86 - 2, s, 3);
    }
  }, [1, 1]);
}

/** Flat roof: gravel with a few patches. One tile = 3 m (world-scale UVs). */
export function roofTexture(): THREE.Texture {
  return canvasTexture(256, (c, s) => {
    const r = rand(51);
    c.fillStyle = "#5d5f60";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 9000; i++) {
      c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.12})` : `rgba(0,0,0,${r() * 0.15})`;
      c.fillRect(r() * s, r() * s, 1.5, 1.5);
    }
  }, [1, 1]);
}

/** Painted plaster for the house walls. One tile = 2 m (world-scale UVs). */
export function plasterTexture(): THREE.Texture {
  return canvasTexture(256, (c, s) => {
    const r = rand(61);
    c.fillStyle = "#ece7de";
    c.fillRect(0, 0, s, s);
    for (let i = 0; i < 7000; i++) {
      c.fillStyle = r() > 0.5 ? `rgba(255,255,255,${r() * 0.12})` : `rgba(60,50,40,${r() * 0.06})`;
      c.fillRect(r() * s, r() * s, 2, 2);
    }
  }, [1, 1]);
}
