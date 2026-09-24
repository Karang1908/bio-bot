import { useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { brainActivity, useStore, type BrainPoints } from "../lib/store";

/**
 * The fly's nervous system as a point cloud: every drawable neuron at its real soma
 * position (sensory neurons at their partners' mean). Dim blue at rest; a neuron glows
 * warm as its activity rises. Hovering a group in the list tints that group.
 */

const vertex = /* glsl */ `
  attribute float activity;
  attribute float grp;
  uniform float uHighlight;
  uniform float uScale;
  varying float vAct;
  varying float vHi;
  void main() {
    vAct = activity;
    vHi = (uHighlight >= 0.0 && abs(grp - uHighlight) < 0.5) ? 1.0 : 0.0;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = (1.1 + activity * 4.5 + vHi * 0.9) * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const fragment = /* glsl */ `
  varying float vAct;
  varying float vHi;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c);
    if (d > 0.25) discard;
    float soft = smoothstep(0.25, 0.0, d);
    vec3 rest = mix(vec3(0.42, 0.55, 0.80), vec3(0.55, 0.85, 1.0), vHi);
    vec3 hot = vec3(1.0, 0.80, 0.38);
    float a = clamp(vAct * 1.8, 0.0, 1.0);
    vec3 col = mix(rest, hot, a);
    float alpha = (0.16 + vHi * 0.25 + a * 0.8) * soft;
    gl_FragColor = vec4(col * alpha, alpha);
  }
`;

function Cloud({ points }: { points: BrainPoints }) {
  const highlight = useStore((s) => s.brainHighlight);
  const dpr = useThree((s) => s.viewport.dpr);

  const { geometry, material, display } = useMemo(() => {
    const n = points.count;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      // data: x left-right, y dorsal-ventral-ish, z brain -> nerve cord. Draw brain on top.
      pos[i * 3] = points.positions[i * 3];
      pos[i * 3 + 1] = -points.positions[i * 3 + 2];
      pos[i * 3 + 2] = -points.positions[i * 3 + 1];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.computeBoundingBox();
    const mid = geo.boundingBox!.getCenter(new THREE.Vector3());
    geo.translate(-mid.x, -mid.y, -mid.z); // centre on the whole CNS, not the brain-heavy mean
    geo.setAttribute("grp", new THREE.BufferAttribute(Float32Array.from(points.groups), 1));
    const display = new Float32Array(n);
    const act = new THREE.BufferAttribute(display, 1);
    act.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("activity", act);
    geo.computeBoundingSphere();
    const mat = new THREE.ShaderMaterial({
      vertexShader: vertex,
      fragmentShader: fragment,
      uniforms: { uHighlight: { value: -1 }, uScale: { value: 2.4 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return { geometry: geo, material: mat, display };
  }, [points]);

  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  useEffect(() => { material.uniforms.uHighlight.value = highlight ?? -1; }, [material, highlight]);
  useEffect(() => { material.uniforms.uScale.value = 2.4 * dpr; }, [material, dpr]);

  useFrame(() => {
    const target = brainActivity.data;
    if (!target || target.length !== display.length) return;
    // Ease toward the latest frame so activity glows and fades instead of flickering.
    for (let i = 0; i < display.length; i++) display[i] += (target[i] / 255 - display[i]) * 0.22;
    geometry.attributes.activity.needsUpdate = true;
  });

  return <points geometry={geometry} material={material} />;
}

export default function BrainView() {
  const points = useStore((s) => s.brainPoints);
  const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return (
    <Canvas dpr={[1, 2]} camera={{ position: [1.05, 0.1, 2.0], fov: 36, near: 0.01, far: 20 }}
      gl={{ antialias: true, alpha: true }} style={{ background: "transparent" }}>
      {points && <Cloud points={points} />}
      <OrbitControls makeDefault enablePan={false} minDistance={0.6} maxDistance={4}
        autoRotate={!reduce} autoRotateSpeed={0.5} target={[0, 0, 0]} />
    </Canvas>
  );
}
