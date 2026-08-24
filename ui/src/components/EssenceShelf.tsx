import { forwardRef } from 'react';
import type { Essence } from '../lib/api';
import { BottleBadge, TrashIcon } from './icons';

// Essence shelf (spec §4.2.1): vertical stack, thumbnail recognizability
// first, the bottle "color" badge is a secondary indicator only.
const EssenceShelf = forwardRef<HTMLDivElement, { essences: Essence[]; onDelete: (essence: Essence) => void }>(
  function EssenceShelf({ essences, onDelete }, ref) {
    function onDragStart(e: React.DragEvent, essence: Essence) {
      e.dataTransfer.setData('text/essence-id', essence.id);
      e.dataTransfer.effectAllowed = 'move';
    }

    function handleDelete(e: React.MouseEvent, essence: Essence) {
      e.stopPropagation();
      if (window.confirm(`Delete the Essence "${essence.name}"? This can't be undone.`)) {
        onDelete(essence);
      }
    }

    return (
      <div ref={ref} className="w-56 shrink-0 border-l border-white/5 bg-charcoal/60 p-4 overflow-y-auto">
        <h2 className="font-display text-ink-soft text-sm tracking-wide mb-3">Essences</h2>
        {essences.length === 0 && (
          <p className="text-ink-soft text-xs leading-relaxed">
            Nothing distilled yet. Visit the Distillation Room to extract one from a reference image.
          </p>
        )}
        <div className="flex flex-col gap-2">
          {essences.map((e) => (
            <div
              key={e.id}
              draggable
              onDragStart={(ev) => onDragStart(ev, e)}
              className="group relative flex items-center gap-2.5 p-2 rounded-card bg-surface/70 hover:bg-surface cursor-grab active:cursor-grabbing transition-colors"
              title={`Drag onto the photo to apply "${e.name}"`}
            >
              <img
                src={e.thumbnail}
                alt=""
                className="w-10 h-10 rounded object-cover shrink-0 border border-white/10"
              />
              <div className="min-w-0 flex-1">
                <p className="text-ink text-xs truncate">{e.name}</p>
              </div>
              <BottleBadge color={e.color} />
              <button
                onClick={(ev) => handleDelete(ev, e)}
                title={`Delete "${e.name}"`}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-charcoal border border-white/10 text-ink-soft hover:text-red-300 hover:border-red-300/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
              >
                <TrashIcon className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      </div>
    );
  },
);

export default EssenceShelf;
