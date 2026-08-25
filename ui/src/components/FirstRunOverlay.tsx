import { useEffect, useState } from 'react';
import type { BootstrapStatus } from '../lib/appBridge';

// Packaged-install first run only: sidecar/venv doesn't exist yet the way
// it always does in dev (prebuilt by `npm run sidecar:setup`) —
// electron/sidecarBootstrap.js builds a private Python runtime from the
// embeddable interpreter bundled into the installer, and this shows that
// progress instead of leaving the window looking frozen for however long
// that takes (pip-installing torch et al. can be a few minutes). Never
// appears in dev — getCurrentBootstrapStatus()/onBootstrapProgress simply
// never fire there. Disappears once the sidecar itself responds to a
// health check; ModelStatusBanner takes over from there for the separate,
// much larger model-download phase.
export default function FirstRunOverlay() {
  const [status, setStatus] = useState<BootstrapStatus | null>(null);
  const [sidecarUp, setSidecarUp] = useState(false);

  useEffect(() => {
    let cancelled = false;

    // Catches the race where main.js already pushed a status before this
    // component mounted — main.js remembers the latest and hands it back.
    window.appBridge.getCurrentBootstrapStatus().then((s) => {
      if (!cancelled && s) setStatus(s);
    });

    const unsubscribe = window.appBridge.onBootstrapProgress((s) => {
      if (!cancelled) setStatus(s);
    });

    let timer: ReturnType<typeof setTimeout>;
    async function pollHealth() {
      try {
        const health = await window.appBridge.getSidecarHealth();
        if (!cancelled && health.ok) {
          setSidecarUp(true);
          return; // stop polling — nothing left for this overlay to do
        }
      } catch {
        // sidecar not up yet (still bootstrapping, or just starting)
      }
      if (!cancelled) timer = setTimeout(pollHealth, 1000);
    }
    pollHealth();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      unsubscribe();
    };
  }, []);

  if (sidecarUp || !status) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-dusk flex flex-col items-center justify-center gap-4 px-8 text-center">
      <div className="w-10 h-10 rounded-full border-2 border-gold/30 border-t-gold animate-spin" />
      <p className="font-display text-ink text-lg">{status.step}</p>
      {status.detail && <p className="text-ink-soft text-xs max-w-md truncate">{status.detail}</p>}
      <p className="text-ink-soft text-[11px] max-w-sm">
        This only happens once — Rasa is setting up its local Python environment and GPU support.
      </p>
    </div>
  );
}
