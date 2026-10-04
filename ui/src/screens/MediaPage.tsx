import { useEffect, useState } from 'react';
import Lightbox from '../components/Lightbox';
import ParallaxImage from '../components/ParallaxImage';
import { DownloadIcon, SunIcon, TrashIcon } from '../components/icons';
import { api, type MediaItem } from '../lib/api';

// Media Page (spec §4.2.3): every finished creation is saved automatically —
// sidecar/app.py's /apply handler persists via media.save_creation on every
// successful application, so this is otherwise a pure read/list view.
export default function MediaPage() {
  const [items, setItems] = useState<MediaItem[] | null>(null);
  const [viewing, setViewing] = useState<MediaItem | null>(null);
  const [animatingIds, setAnimatingIds] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState<{ text: string; gifPath?: string } | null>(null);

  useEffect(() => {
    api
      .listMedia()
      .then(setItems)
      .catch(() => setItems([]));
  }, []);

  async function handleDownload(item: MediaItem) {
    try {
      const saved = await window.appBridge.saveImageDataUrl(item.image, `rasa-${item.essence_name || 'result'}`.replace(/[^\w-]+/g, '_'));
      if (saved) {
        setStatus({ text: 'Saved.', gifPath: saved });
      }
    } catch {
      setStatus({ text: 'Could not save image.' });
    }
  }

  async function handleDelete(item: MediaItem) {
    if (!window.confirm('Delete this creation? This can\'t be undone.')) return;
    try {
      await api.deleteMedia(item.id);
      setItems((prev) => prev?.filter((i) => i.id !== item.id) ?? prev);
    } catch {
      // best-effort — leave the item in place so the user can retry
    }
  }

  async function handleAnimate(item: MediaItem) {
    if (animatingIds.has(item.id)) return;
    setAnimatingIds((prev) => new Set(prev).add(item.id));
    setStatus({ text: `Animating "${item.essence_name}"…` });
    try {
      const result = await api.generateGif(item.id);
      setItems((prev) => prev?.map((i) => (i.id === item.id ? { ...i, has_gif: true } : i)) ?? prev);
      setStatus({ text: 'GIF ready.', gifPath: result.gif_path });
    } catch (err) {
      setStatus({ text: err instanceof Error ? err.message : 'Animation failed.' });
      setTimeout(() => setStatus(null), 3000);
    } finally {
      setAnimatingIds((prev) => {
        const next = new Set(prev);
        next.delete(item.id);
        return next;
      });
    }
  }

  return (
    <div className="h-full overflow-y-auto p-8">
      <h1 className="font-display text-2xl text-ink mb-6">Media</h1>

      {items === null && <p className="text-ink-soft">Loading…</p>}
      {items?.length === 0 && <p className="text-ink-soft">Nothing created yet — apply an Essence on the Main Stage.</p>}

      <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-4">
        {items?.map((item) => (
          <div key={item.id} className="group relative bg-surface/60 rounded-card overflow-hidden">
            <button onClick={() => setViewing(item)} title="View full size" className="block w-full">
              <ParallaxImage src={item.image} depthSrc={item.depth} fit="cover" className="w-full aspect-square" />
            </button>
            <div className="p-2.5">
              <p className="text-ink text-xs truncate">{item.essence_name}</p>
              <p className="text-ink-soft text-[11px]">{new Date(item.created_at).toLocaleString()}</p>
            </div>
            <button
              onClick={() => handleDelete(item)}
              title="Delete"
              className="absolute top-2 right-2 w-7 h-7 rounded-full bg-charcoal/90 border border-white/10 text-ink-soft hover:text-red-300 hover:border-red-300/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
            >
              <TrashIcon className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => handleDownload(item)}
              title="Download image"
              className="absolute top-2 right-11 w-7 h-7 rounded-full bg-charcoal/90 border border-white/10 text-ink-soft hover:text-gold hover:border-gold/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
            >
              <DownloadIcon className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => handleAnimate(item)}
              disabled={animatingIds.has(item.id)}
              title={item.has_gif ? 'Re-animate (sweeping-light GIF)' : 'Animate (sweeping-light GIF)'}
              className={`absolute top-2 left-2 w-7 h-7 rounded-full bg-charcoal/90 border flex items-center justify-center transition-colors ${
                item.has_gif ? 'border-gold/60 text-gold opacity-100' : 'border-white/10 text-ink-soft opacity-0 group-hover:opacity-100'
              } ${animatingIds.has(item.id) ? 'animate-pulse' : 'hover:text-gold hover:border-gold/40'}`}
            >
              <SunIcon className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>

      {viewing && (
        <Lightbox src={viewing.image} depthSrc={viewing.depth} alt={viewing.essence_name} onClose={() => setViewing(null)} />
      )}

      {status && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-charcoal/90 text-ink-soft text-sm px-4 py-2 rounded-full border border-white/10 flex items-center gap-3 z-40">
          <span>{status.text}</span>
          {status.gifPath && (
            <button
              onClick={() => {
                window.appBridge.showInFolder(status.gifPath!);
                setStatus(null);
              }}
              className="text-gold hover:underline"
            >
              Show in folder
            </button>
          )}
          <button onClick={() => setStatus(null)} className="text-ink-soft hover:text-ink">
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
