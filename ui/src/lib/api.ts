// Typed wrappers around window.appBridge.sidecarCall for the essence/media
// endpoints in sidecar/app.py. Keeps the sidecar's HTTP shape out of components.

export interface Essence {
  id: string;
  name: string;
  technique: string;
  created_at: string;
  color: [number, number, number];
  thumbnail: string; // data: URL
}

export interface ApplyResult {
  steps: string[]; // data: URLs, original -> final
  final: string;
  media_id: string;
}

export interface MediaItem {
  id: string;
  essence_id: string;
  essence_name: string;
  created_at: string;
  image: string; // data: URL
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await window.appBridge.sidecarCall(method, path, body);
  if (!res.ok) throw new Error(res.error ?? `Sidecar call failed: ${method} ${path}`);
  return res.data as T;
}

export const api = {
  health: () => window.appBridge.getSidecarHealth(),

  extractEssence: (imagePath: string, name?: string) =>
    call<Essence>('POST', '/essences/extract', { image_path: imagePath, name }),

  listEssences: () => call<{ essences: Essence[] }>('GET', '/essences').then((r) => r.essences),

  deleteEssence: (essenceId: string) => call<{ ok: true }>('DELETE', `/essences/${essenceId}`),

  applyEssence: (essenceId: string, imagePath: string, steps = 8) =>
    call<ApplyResult>('POST', '/apply', { essence_id: essenceId, image_path: imagePath, steps }),

  listMedia: () => call<{ media: MediaItem[] }>('GET', '/media').then((r) => r.media),

  deleteMedia: (mediaId: string) => call<{ ok: true }>('DELETE', `/media/${mediaId}`),
};
