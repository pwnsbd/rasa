import { useEffect, useState } from 'react';
import { api } from '../lib/api';

// First run downloads ~9GB (SDXL base + InstantStyle IP-Adapter) before
// extraction/application can work at all — sidecar/app.py's endpoints
// fail fast with a 503 until it's ready, so without this the Distillation
// Room's shimmer would just sit there with no explanation for several
// minutes. Polls /models/status and shows nothing once state is 'ready'.
export default function ModelStatusBanner() {
  const [status, setStatus] = useState<{ state: string; detail: string | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const s = await api.modelStatus();
        if (!cancelled) setStatus(s);
      } catch {
        // sidecar not up yet
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    }
    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  if (!status || status.state === 'ready') return null;

  const isError = status.state === 'error';

  return (
    <div
      className={`fixed top-0 left-16 right-0 z-50 px-4 py-2 text-xs flex items-center gap-2 border-b ${
        isError ? 'bg-red-950/80 border-red-500/30 text-red-200' : 'bg-charcoal/95 border-white/10 text-ink-soft'
      }`}
    >
      {!isError && <span className="w-2 h-2 rounded-full bg-gold animate-pulse shrink-0" />}
      <span>
        {isError
          ? `Style model failed to load: ${status.detail}`
          : status.detail || 'Loading style model…'}
      </span>
    </div>
  );
}
