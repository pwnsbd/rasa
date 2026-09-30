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

// Essence-aware starting point for the intensity slider, in the same spirit
// as generation.py's suggest_subject_params (segmentation.py) for the
// subject-region strength: a measured property of the *specific* thing
// being applied suggests a better default than one flat number for every
// essence, while staying a suggestion the user is always free to drag away
// from (see MainStage.tsx: only used until the slider is touched).
//
// Uses texture.roughness + texture.high_frequency_energy (both already
// extracted at Distillation time — see style_analysis/texture.py, both
// normalized 0..1) as a "busyness" score. A busy/high-frequency essence
// (dense brushstrokes, film grain, heavy texture) suggests a *lower*
// intensity so the Tile ControlNet holds the photo's own structure more
// tightly and the essence doesn't overwhelm it; a smooth/flat essence
// (soft painterly wash, gentle gradient) can push higher before it reads as
// too strong. A neutral busyness of 0.5 resolves to exactly DEFAULT_INTENSITY,
// and an essence with no texture analysis (blended, or pre-schema) falls
// back to DEFAULT_INTENSITY too — this is a nudge around the existing
// default, not a replacement for it.
const BUSYNESS_SWING = 0.3;

export function suggestIntensity(essence: { analysis?: { texture?: { roughness: number; high_frequency_energy: number } | null } }): number {
  const texture = essence.analysis?.texture;
  if (!texture) return DEFAULT_INTENSITY;
  const busyness = (texture.roughness + texture.high_frequency_energy) / 2;
  const suggested = DEFAULT_INTENSITY - (busyness - 0.5) * BUSYNESS_SWING;
  return Math.min(1, Math.max(0, suggested));
}

// Quality/speed toggle for step count — a real trade-off (more steps ~=
// more time, generally crisper detail), left as a coarse two-way choice
// rather than its own raw slider since step count doesn't have the same
// "which direction is more style" ambiguity strength/controlnet_scale did.
export const STEPS_STANDARD = 30; // sidecar's existing default — unchanged baseline
export const STEPS_HIGH_DETAIL = 45;
