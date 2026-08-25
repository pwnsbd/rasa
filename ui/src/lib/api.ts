// Typed wrappers around window.appBridge.sidecarCall for the essence/media
// endpoints in sidecar/app.py. Keeps the sidecar's HTTP shape out of components.

export interface BlendIngredientInfo {
  name: string;
  weight: number; // normalized share of the blend, 0..1
  source: 'image' | 'essence';
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

export interface ApplyOptions {
  strength?: number;
  controlnetScale?: number;
  steps?: number;
  preserveColor?: boolean;
  blendMode?: BlendMode;
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
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await window.appBridge.sidecarCall(method, path, body);
  if (!res.ok) throw new Error(res.error ?? `Sidecar call failed: ${method} ${path}`);
  return res.data as T;
}

export const api = {
  health: () => window.appBridge.getSidecarHealth(),

  modelStatus: () => call<{ state: 'idle' | 'loading' | 'ready' | 'error'; detail: string | null }>('GET', '/models/status'),

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
      ...(options.preserveColor !== undefined ? { preserve_color: options.preserveColor } : {}),
      ...(options.blendMode !== undefined ? { blend_mode: options.blendMode } : {}),
    }),

  listMedia: () => call<{ media: MediaItem[] }>('GET', '/media').then((r) => r.media),

  deleteMedia: (mediaId: string) => call<{ ok: true }>('DELETE', `/media/${mediaId}`),
};
