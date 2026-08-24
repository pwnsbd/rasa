import { useEffect, useState } from 'react';
import { TrashIcon } from '../components/icons';
import { api, type MediaItem } from '../lib/api';

// Media Page (spec §4.2.3): every finished creation is saved automatically —
// sidecar/app.py's /apply handler persists via media.save_creation on every
// successful application, so this is otherwise a pure read/list view.
export default function MediaPage() {
  const [items, setItems] = useState<MediaItem[] | null>(null);

  useEffect(() => {
    api
      .listMedia()
      .then(setItems)
      .catch(() => setItems([]));
  }, []);

  async function handleDelete(item: MediaItem) {
    if (!window.confirm('Delete this creation? This can\'t be undone.')) return;
    try {
      await api.deleteMedia(item.id);
      setItems((prev) => prev?.filter((i) => i.id !== item.id) ?? prev);
    } catch {
      // best-effort — leave the item in place so the user can retry
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
            <img src={item.image} alt="" className="w-full aspect-square object-cover" />
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
          </div>
        ))}
      </div>
    </div>
  );
}
