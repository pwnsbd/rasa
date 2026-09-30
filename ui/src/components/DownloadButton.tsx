import { useEffect, useRef, useState } from 'react';
import { api, type DownloadStatus } from '../lib/api';
import { DownloadIcon } from './icons';

// Browser-style download indicator, top-right corner. Model weights (~13GB)
// download in the background from sidecar startup (sidecar/model_downloads.py);
// this shows overall progress as a ring around the icon and a popover with
// per-model progress, speed/ETA, and pause/resume. Hidden once everything is
// downloaded. Downloads only run while the app is open and resume next launch.

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

function fmtEta(s: number | null): string {
  if (s == null) return '';
  if (s < 90) return `${Math.round(s)}s left`;
  if (s < 5400) return `${Math.round(s / 60)} min left`;
  return `${(s / 3600).toFixed(1)} h left`;
}

export default function DownloadButton() {
  const [st, setSt] = useState<DownloadStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const s = await api.downloadStatus();
        if (!cancelled) setSt(s);
      } catch {
        // sidecar not up yet
      }
      if (!cancelled) timer = setTimeout(poll, 1000);
    }
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  if (!st || st.state === 'ready' || st.state === 'idle') return null;

  const pct = st.total_bytes ? Math.min(1, st.downloaded_bytes / st.total_bytes) : 0;
  const isError = st.state === 'error';
  const isPaused = st.state === 'paused';
  const R = 10;
  const C = 2 * Math.PI * R;

  async function act(fn: () => Promise<DownloadStatus>) {
    setBusy(true);
    try { setSt(await fn()); } catch { /* next poll will reflect state */ }
    setBusy(false);
  }

  return (
    <div ref={rootRef} className="fixed top-3 right-4 z-[60]">
      <button
        onClick={() => setOpen((o) => !o)}
        title={isError ? 'Download failed' : isPaused ? 'Downloads paused' : `Downloading models… ${Math.round(pct * 100)}%`}
        className={`relative w-9 h-9 rounded-full bg-charcoal/95 border flex items-center justify-center hover:bg-charcoal ${
          isError ? 'border-red-500/50 text-red-300' : 'border-white/10 text-gold'
        }`}
      >
        <svg viewBox="0 0 24 24" className="absolute inset-0 w-full h-full -rotate-90">
          <circle cx="12" cy="12" r={R} fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1.6" />
          <circle
            cx="12" cy="12" r={R} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
            strokeDasharray={C} strokeDashoffset={C * (1 - pct)}
          />
        </svg>
        <DownloadIcon className={`w-4 h-4 ${st.state === 'downloading' ? 'animate-pulse' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-80 rounded-xl bg-charcoal border border-white/10 shadow-xl p-4 text-xs text-ink-soft">
          <div className="flex items-baseline justify-between mb-1">
            <span className="font-display text-ink text-sm">Models</span>
            <span>{fmtBytes(st.downloaded_bytes)} / {fmtBytes(st.total_bytes)}</span>
          </div>
          <div className="h-1.5 rounded-full bg-white/10 overflow-hidden mb-1">
            <div className="h-full bg-gold transition-all" style={{ width: `${pct * 100}%` }} />
          </div>
          <p className="mb-3">
            {isError ? `Failed: ${st.error}` :
              isPaused ? 'Paused' :
              `${fmtBytes(st.speed_bps)}/s ${fmtEta(st.eta_seconds)}`}
          </p>

          <ul className="space-y-2 mb-3">
            {st.items.map((it) => {
              const p = it.state === 'done' ? 1 : it.total_bytes ? it.downloaded_bytes / it.total_bytes : 0;
              return (
                <li key={it.id}>
                  <div className="flex justify-between">
                    <span className="text-ink">{it.label}</span>
                    <span>{it.state === 'done' ? 'Done' : it.state === 'pending' ? 'Waiting' : `${Math.round(p * 100)}%`}</span>
                  </div>
                  {it.state !== 'done' && it.state !== 'pending' && (
                    <div className="h-1 rounded-full bg-white/10 overflow-hidden mt-1">
                      <div className="h-full bg-gold/70" style={{ width: `${p * 100}%` }} />
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px]">Downloads continue while Rasa is open and resume next launch. Extract/Apply unlock when done.</span>
            {st.state === 'downloading' && (
              <button disabled={busy} onClick={() => act(api.pauseDownloads)} className="shrink-0 rounded-lg border border-white/15 px-3 py-1 text-ink hover:bg-white/5">Pause</button>
            )}
            {(isPaused || isError) && (
              <button disabled={busy} onClick={() => act(api.resumeDownloads)} className="shrink-0 rounded-lg bg-gold px-3 py-1 text-dusk">{isError ? 'Retry' : 'Resume'}</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
