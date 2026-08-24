import { useEffect } from 'react';

// Full-size image viewer overlay. Click the backdrop, press Escape, or hit
// the close button to dismiss.
export default function Lightbox(props: { src: string; alt?: string; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') props.onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [props.onClose]);

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
      <img
        src={props.src}
        alt={props.alt ?? ''}
        onClick={(e) => e.stopPropagation()}
        className="max-w-full max-h-full object-contain rounded-card shadow-2xl"
      />
    </div>
  );
}
