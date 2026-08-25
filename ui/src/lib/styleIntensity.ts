// Style intensity -> strength/controlnet_scale translation layer (the
// "Main Stage artistic controls" + parameter-translation-layer work,
// previously deferred). The sidecar's real knobs are `strength` (how far
// img2img denoising pushes from the original) and `controlnet_scale` (how
// tightly the Tile ControlNet holds the original's structure) — they need
// to move in *opposite* directions together to produce "more/less visible
// style" as one intuitive control, rather than being exposed as two raw
// sliders a user has to understand the interaction between (see
// sidecar/generation.py's DEFAULT_STRENGTH comment for why).
//
// intensity=0.5 reproduces generation.py's own defaults exactly
// (strength=0.85, controlnet_scale=0.85), so a user who never touches the
// slider gets identical behavior to before this control existed.
export interface StyleParams {
  strength: number;
  controlnetScale: number;
}

export function intensityToParams(intensity: number): StyleParams {
  const t = Math.min(1, Math.max(0, intensity));
  return {
    strength: 0.75 + 0.2 * t,
    controlnetScale: 0.95 - 0.2 * t,
  };
}

export const DEFAULT_INTENSITY = 0.5;

// Quality/speed toggle for step count — a real trade-off (more steps ~=
// more time, generally crisper detail), left as a coarse two-way choice
// rather than its own raw slider since step count doesn't have the same
// "which direction is more style" ambiguity strength/controlnet_scale did.
export const STEPS_STANDARD = 30; // sidecar's existing default — unchanged baseline
export const STEPS_HIGH_DETAIL = 45;
