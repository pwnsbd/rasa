import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import DistillationBottle from '../components/DistillationBottle';
import { BottleBadge } from '../components/icons';
import { api, type Essence } from '../lib/api';

type Phase = 'idle' | 'shimmering' | 'pouring' | 'sealed';

// Distillation Room (spec §4.2.2): a focused space separate from the Main
// Stage. Extracting doesn't dump the essence onto the shelf automatically —
// it only appears there once the user navigates away, which App.tsx gets
// for free by only mounting the active zone (MainStage refetches on mount).
export default function DistillationRoom() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [fill, setFill] = useState(0);
  const [pendingEssence, setPendingEssence] = useState<Essence | null>(null);
  const [sessionEssences, setSessionEssences] = useState<Essence[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fillState = useRef({ v: 0 });

  // Dev-only automation hook — see appBridge.d.ts / MainStage.tsx.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    window.__testHooks = {
      ...window.__testHooks,
      distillation: {
        chooseReferencePath: (path: string) => chooseReference(path),
        seal: () => seal(),
        state: () => ({ phase, sessionCount: sessionEssences.length, pendingName: pendingEssence?.name }),
      },
    };
  }, [phase, sessionEssences, pendingEssence]);

  async function chooseReference(overridePath?: string) {
    const path = overridePath ?? (await window.appBridge.openImageDialog());
    if (!path) return;
    setError(null);
    setPreviewUrl(await window.appBridge.readImageAsDataUrl(path));
    setPhase('shimmering');
    setFill(0);

    try {
      const result = await api.extractEssence(path);
      setPendingEssence(result);
      setPhase('pouring');
      pourIn();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed.');
      setPhase('idle');
    }
  }

  function pourIn() {
    fillState.current.v = 0;
    gsap.to(fillState.current, {
      v: 1,
      duration: 1.8,
      ease: 'sine.inOut',
      onUpdate: () => setFill(fillState.current.v),
      onComplete: () => setPhase('sealed'),
    });
  }

  function seal() {
    if (!pendingEssence) return;
    setSessionEssences((prev) => [pendingEssence, ...prev]);
    setPendingEssence(null);
    setPreviewUrl(null);
    setFill(0);
    setPhase('idle');
  }

  return (
    <div className="h-full flex flex-col items-center justify-center gap-8 p-10">
      <h1 className="font-display text-2xl text-ink">Distillation Room</h1>

      {phase === 'idle' && (
        <button
          onClick={() => chooseReference()}
          className="border border-dashed border-white/15 rounded-card px-10 py-16 text-ink-soft hover:text-ink hover:border-gold/50 transition-colors font-body"
        >
          Drop in a reference photo to distill its Essence
        </button>
      )}

      {phase !== 'idle' && (
        <div className="flex items-center gap-12">
          <div className="relative w-56 h-56 rounded-card overflow-hidden border border-white/10">
            {previewUrl && <img src={previewUrl} alt="Reference" className="w-full h-full object-cover" />}
            {phase === 'shimmering' && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent shimmer-sweep" />
            )}
          </div>

          <div className="flex flex-col items-center gap-4">
            <DistillationBottle fill={fill} color={pendingEssence?.color ?? [201, 163, 92]} />
            <p className="text-ink-soft text-sm">
              {phase === 'shimmering' && 'Shimmering…'}
              {phase === 'pouring' && 'Distilling…'}
              {phase === 'sealed' && pendingEssence?.name}
            </p>
            {phase === 'sealed' && (
              <div className="flex gap-2">
                <button
                  onClick={seal}
                  className="bg-gold/90 hover:bg-gold text-charcoal text-sm px-4 py-1.5 rounded-full font-medium"
                >
                  Seal &amp; distill another
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {error && <p className="text-red-300 text-sm">{error}</p>}

      {sessionEssences.length > 0 && (
        <div className="mt-4 pt-6 border-t border-white/5 w-full max-w-2xl">
          <p className="text-ink-soft text-xs mb-3 text-center">Sealed this session — leave the room to add these to the shelf</p>
          <div className="flex flex-wrap justify-center gap-3">
            {sessionEssences.map((e) => (
              <div key={e.id} className="flex items-center gap-2 bg-surface/60 rounded-full px-3 py-1.5">
                <BottleBadge color={e.color} size={16} />
                <span className="text-ink text-xs">{e.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
