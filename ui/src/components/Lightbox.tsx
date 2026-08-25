import { useEffect, useState } from 'react';
import ParallaxImage from './ParallaxImage';

const VIEWPORT_PADDING = 128; // roughly matches the overlay's p-8 plus breathing room

// Full-size image viewer overlay. Click the backdrop, press Escape, or hit
// the close button to dismiss.
//
// Sizing is computed explicitly in JS (natural image size, clamped to the
// viewport, preserving aspect ratio) rather than relying on a plain <img>'s
// "replaced element" max-width/max-height/object-fit CSS behavior — a
// wrapping <div> (what ParallaxImage needs for its canvas overlay) doesn't
// get that same special sizing, so this computes the same result the old
// single <img> used to get for free. Falls back to the original plain <img>
// + CSS approach while the natural size is still loading, so there's no
// visible flash/resize once ParallaxImage takes over.
export default function Lightbox(props: { src: string; depthSrc?: string | null; alt?: string; onClose: () => void }) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });

  useEffect(() => {
    setNatural(null);
    const img = new Image();
    img.onload = () => setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = props.src;
  }, [props.src]);

  useEffect(() => {
    function onResize() {
      setViewport({ w: window.innerWidth, h: window.innerHeight });
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') props.onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [props.onClose]);

  const availW = Math.max(100, viewport.w - VIEWPORT_PADDING);
  const availH = Math.max(100, viewport.h - VIEWPORT_PADDING);
  const size = natural
    ? (() => {
        const scale = Math.min(1, availW / natural.w, availH / natural.h);
        return { width: natural.w * scale, height: natural.h * scale };
      })()
    : null;

  return (
    <div
      className="fixed inset-0 z-50 bg-charcoal/90 backdrop-blur-sm flex items-center justify-center p-8"
      onClick={props.onClose}
    >
      <button
        onClick={props.onClose}
        title="Close"
        className="absolute top-5 right-5 w-9 h-9 rounded-full bg-surface/80 border border-white/10 text-ink-soft hover:text-ink flex items-center justify-center"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" className="w-4 h-4">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
      {size ? (
        <ParallaxImage
          src={props.src}
          depthSrc={props.depthSrc}
          alt={props.alt ?? ''}
          fit="contain"
          className="rounded-card shadow-2xl"
          style={{ width: size.width, height: size.height }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <img
          src={props.src}
          alt={props.alt ?? ''}
          onClick={(e) => e.stopPropagation()}
          className="max-w-full max-h-full object-contain rounded-card shadow-2xl"
        />
      )}
    </div>
  );
}
