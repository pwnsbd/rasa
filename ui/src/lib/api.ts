// Typed wrappers around window.appBridge.sidecarCall for the essence/media
// endpoints in sidecar/app.py. Keeps the sidecar's HTTP shape out of components.

export interface BlendIngredientInfo {
  name: string;
  weight: number; // normalized share of the blend, 0..1
  source: 'image' | 'essence';
}

export interface ColorSwatch {
  hex: string;
  rgb: [number, number, number];
  weight: number; // relative share of the image, 0..1
}

export interface PaletteProfile {
  dominant_colors: ColorSwatch[];
  mean_saturation: number;
  mean_luminance: number;
  temperature: number; // -1 (cool) .. 1 (warm)
  contrast: number;
}

export interface TextureProfile {
  roughness: number;
  detail_density: number;
  high_frequency_energy: number;
  repetition: number;
}

export interface StrokeProfile {
  directionality: number; // 0 (no dominant direction) .. 1 (strongly directional)
  curvature: number;
  coherence: number;
  density: number;
  dominant_angle: number | null; // radians, (-pi/2, pi/2] — the essence's aggregate stroke axis; null when directionality is too low for one to be meaningful
  orientation_map_path: string | null;
  field_path: string | null; // raw per-cell orientation field on disk — presence (not the path itself) is what tells the UI a stroke-texture apply is possible for this essence
}

export interface StyleStatistics {
  abstraction: number | null;
  edge_density: number;
  local_contrast: number;
}

// Structured style analysis (sidecar/style_analysis/), computed once at
// Distillation time and stored on the Essence. Any field can be null: a
// blended (Cauldron) Essence has no single reference to analyze, an
// essence saved before this schema existed has none of it, and stroke
// analysis specifically can fail independently of the others (see
// essence_models.py's EssenceMeta docstring).
export interface EssenceAnalysis {
  palette: PaletteProfile | null;
  texture: TextureProfile | null;
  stroke: StrokeProfile | null;
  style_statistics: StyleStatistics | null;
}

export interface Essence {
  id: string;
  name: string;
  technique: string;
  created_at: string;
  color: [number, number, number];
  thumbnail: string; // data: URL
  // Present only on an Essence created via the Cauldron.
  blended_from?: BlendIngredientInfo[] | null;
  analysis: EssenceAnalysis;
  // Whether a tileable material swatch (texture_source.png) was persisted
  // for this essence — same file-existence pattern as MediaItem's has_gif
  // below. False for a blended (Cauldron) Essence or one saved before this
  // existed; the "texture_overlay" apply mode needs this to be true.
  has_texture_source: boolean;
}

// One ingredient sent to POST /essences/blend — either a fresh photo or an
// existing Essence, each with a relative weight (not required to sum to 1;
// the sidecar normalizes).
export type BlendIngredientInput =
  | { type: 'image'; imagePath: string; weight: number }
  | { type: 'essence'; essenceId: string; weight: number };

export interface ApplyResult {
  steps: string[]; // data: URLs, original -> final
  final: string;
  media_id: string;
}

export type BlendMode = 'subject' | 'depth' | 'none';

// "restyle" (default): the real SDXL/IP-Adapter/ControlNet diffusion
// pipeline. "texture_overlay": a second, non-diffusion engine that
// composites the essence's own material swatch onto the target as a
// classical bump/emboss overlay, preserving the target's content with
// total fidelity — see sidecar/texture_overlay.py and
// generation.apply_essence's own docstring for why this exists as a
// separate engine rather than another restyle slider.
export type ApplyMode = 'restyle' | 'texture_overlay';

export interface ApplyOptions {
  strength?: number;
  controlnetScale?: number;
  steps?: number;
  colorPreservation?: number; // 0..1 — see sidecar/color_transfer.py
  strokeAmount?: number; // 0..1 — see sidecar/stroke_texture.py; no-op when the essence has no stroke field
  blendMode?: BlendMode;
  mode?: ApplyMode;
  textureOverlayAmount?: number; // 0..1, only used when mode === 'texture_overlay'
}

export interface MediaItem {
  id: string;
  essence_id: string;
  essence_name: string;
  created_at: string;
  image: string; // data: URL
  // Depth map computed at apply time (see sidecar/generation.py's
  // compute_depth) — drives ParallaxImage's hover effect. null for
  // creations made before this existed, or with compute_depth=False.
  depth?: string | null;
  // Whether a relight GIF (see sidecar/relight.py) has already been
  // generated and saved for this creation.
  has_gif?: boolean;
}

export interface GifResult {
  gif: string; // data: URL
  gif_path: string; // absolute path on disk, for showInFolder
}

export interface DownloadItem {
  id: string;
  label: string;
  total_bytes: number;
  downloaded_bytes: number;
  state: 'pending' | 'downloading' | 'done' | 'error';
}

export interface DownloadStatus {
  state: 'idle' | 'downloading' | 'paused' | 'ready' | 'error';
  error: string | null;
  total_bytes: number;
  downloaded_bytes: number;
  speed_bps: number;
  eta_seconds: number | null;
  items: DownloadItem[];
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await window.appBridge.sidecarCall(method, path, body);
  if (!res.ok) throw new Error(res.error ?? `Sidecar call failed: ${method} ${path}`);
  return res.data as T;
}

export const api = {
  health: () => window.appBridge.getSidecarHealth(),

  modelStatus: () => call<{ state: 'idle' | 'loading' | 'ready' | 'error'; detail: string | null }>('GET', '/models/status'),

  downloadStatus: () => call<DownloadStatus>('GET', '/models/downloads'),
  pauseDownloads: () => call<DownloadStatus>('POST', '/models/downloads/pause'),
  resumeDownloads: () => call<DownloadStatus>('POST', '/models/downloads/resume'),

  extractEssence: (imagePath: string, name?: string) =>
    call<Essence>('POST', '/essences/extract', { image_path: imagePath, name }),

  listEssences: () => call<{ essences: Essence[] }>('GET', '/essences').then((r) => r.essences),

  // The Cauldron: blend existing Essences and/or fresh reference photos
  // into one new Essence. Same response shape as extractEssence.
  blendEssences: (ingredients: BlendIngredientInput[], name?: string) =>
    call<Essence>('POST', '/essences/blend', {
      name,
      ingredients: ingredients.map((ing) =>
        ing.type === 'image'
          ? { type: 'image', image_path: ing.imagePath, weight: ing.weight }
          : { type: 'essence', essence_id: ing.essenceId, weight: ing.weight },
      ),
    }),

  deleteEssence: (essenceId: string) => call<{ ok: true }>('DELETE', `/essences/${essenceId}`),

  // strength/controlnetScale/steps all omitted -> sidecar's own defaults
  // (tuned for the real diffusion pipeline in sidecar/generation.py). The
  // Main Stage computes strength/controlnetScale from its single intensity
  // slider via lib/styleIntensity.ts rather than sending raw values a user
  // never directly sets.
  applyEssence: (essenceId: string, imagePath: string, options: ApplyOptions = {}) =>
    call<ApplyResult>('POST', '/apply', {
      essence_id: essenceId,
      image_path: imagePath,
      ...(options.steps ? { steps: options.steps } : {}),
      ...(options.strength !== undefined ? { strength: options.strength } : {}),
      ...(options.controlnetScale !== undefined ? { controlnet_scale: options.controlnetScale } : {}),
      ...(options.colorPreservation !== undefined ? { color_preservation: options.colorPreservation } : {}),
      ...(options.strokeAmount !== undefined ? { stroke_amount: options.strokeAmount } : {}),
      ...(options.blendMode !== undefined ? { blend_mode: options.blendMode } : {}),
      ...(options.mode !== undefined ? { mode: options.mode } : {}),
      ...(options.textureOverlayAmount !== undefined ? { texture_overlay_amount: options.textureOverlayAmount } : {}),
    }),

  listMedia: () => call<{ media: MediaItem[] }>('GET', '/media').then((r) => r.media),

  deleteMedia: (mediaId: string) => call<{ ok: true }>('DELETE', `/media/${mediaId}`),

  // Relight GIF export (sidecar/relight.py) — a sweeping-light animated
  // loop baked from the creation's image + depth map, computing the depth
  // map on demand server-side if this creation doesn't already have one.
  generateGif: (mediaId: string) => call<GifResult>('POST', `/media/${mediaId}/gif`),
};
