import { useEffect, useState } from 'react';
import type { SidecarHealth } from '../lib/appBridge';

// Minimal placeholder for the Settings zone (spec §4.2.4). Full model-config
// and alchemical-iconography treatment comes later — this exists right now
// to prove the Electron -> sidecar -> GPU-detection path works end to end.
export default function Settings() {
  const [health, setHealth] = useState<SidecarHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    window.appBridge.getSidecarHealth().then((h) => {
      if (!cancelled) {
        setHealth(h);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="max-w-xl mx-auto p-8 font-body">
      <h1 className="font-display text-2xl text-ink mb-6">Device status</h1>

      {loading && <p className="text-ink-soft">Checking sidecar…</p>}

      {!loading && (!health || !health.ok) && (
        <p className="text-ink-soft">
          Sidecar unreachable{health?.error ? `: ${health.error}` : ''}. Run{' '}
          <code className="text-gold">npm run sidecar:setup</code> then <code className="text-gold">npm run dev</code>.
        </p>
      )}

      {!loading && health?.ok && (
        <div className="bg-surface rounded-card p-5 space-y-2">
          <Row label="GPU" value={health.gpu.device_name ?? 'None detected (CPU fallback)'} />
          <Row label="CUDA available" value={String(health.gpu.cuda_available)} />
          <Row label="Compute capability" value={health.gpu.compute_capability ?? '—'} />
          <Row label="VRAM" value={health.gpu.vram_gb ? `${health.gpu.vram_gb} GB` : '—'} />
          <Row label="Torch" value={health.gpu.torch_version ?? '—'} />
          {health.gpu.warning && (
            <p className="mt-3 text-gold text-sm">{health.gpu.warning}</p>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-ink-soft">{label}</span>
      <span className="text-ink">{value}</span>
    </div>
  );
}
