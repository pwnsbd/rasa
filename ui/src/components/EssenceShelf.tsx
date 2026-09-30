import { forwardRef } from 'react';
import type { Essence, EssenceAnalysis } from '../lib/api';
import { BottleBadge, TrashIcon } from './icons';

// Small, secondary "style fingerprint" from the structured analysis
// computed at Distillation time (sidecar/style_analysis/) — palette
// swatches (top dominant colors) + a texture-roughness bar, so a user can
// tell at a glance why one essence applies "louder" than another before
// they drag it, instead of finding out only after generating. Deliberately
// minimal, same spirit as the bottle badge already being secondary to the
// thumbnail (spec §4.2.1): a couple of dots and a thin bar, not a chart.
// Renders nothing for a blended (Cauldron) or pre-schema essence, where
// this analysis doesn't exist (see essence_models.py).
// A short rotated line indicating the essence's dominant stroke axis
// (sidecar/style_analysis/stroke.py's dominant_angle) — shown only when
// there's both an angle confident enough to be meaningful *and* a
// persisted raw field (field_path) for the Main Stage's "Stroke grain"
// control (sidecar/stroke_texture.py) to actually use. Doubles as a hint:
// an essence without this glyph won't respond to that slider at all.
function StrokeGlyph({ stroke }: { stroke: NonNullable<EssenceAnalysis['stroke']> }) {
  if (!stroke.field_path || stroke.dominant_angle === null) return null;
  const deg = (stroke.dominant_angle * 180) / Math.PI;
  return (
    <span
      className="shrink-0 text-gold/70"
      title={`Stroke direction detected (${Math.round(stroke.directionality * 100)}% directional) — supports the Stroke grain control`}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" className="block">
        <line x1="2" y1="6" x2="10" y2="6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" transform={`rotate(${deg} 6 6)`} />
      </svg>
    </span>
  );
}

function EssenceFingerprint({ analysis }: { analysis: EssenceAnalysis }) {
  const swatches = analysis.palette?.dominant_colors.slice(0, 4) ?? [];
  const roughness = analysis.texture?.roughness ?? null;
  const stroke = analysis.stroke;
  if (swatches.length === 0 && roughness === null && !stroke) return null;

  return (
    <div className="flex items-center gap-1.5 mt-1">
      {swatches.length > 0 && (
        <div className="flex -space-x-1 shrink-0">
          {swatches.map((s, i) => (
            <span
              key={i}
              className="w-2.5 h-2.5 rounded-full border border-charcoal"
              style={{ backgroundColor: s.hex }}
              title={`${s.hex} — ${Math.round(s.weight * 100)}% of the reference`}
            />
          ))}
        </div>
      )}
      {roughness !== null && (
        <div
          className="flex-1 h-1 min-w-[20px] rounded-full bg-white/10 overflow-hidden"
          title={`Texture roughness: ${Math.round(roughness * 100)}% — how strongly this essence's own texture reads once applied`}
        >
          <div className="h-full bg-gold/50" style={{ width: `${Math.round(roughness * 100)}%` }} />
        </div>
      )}
      {stroke && <StrokeGlyph stroke={stroke} />}
    </div>
  );
}

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
                <EssenceFingerprint analysis={e.analysis} />
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
