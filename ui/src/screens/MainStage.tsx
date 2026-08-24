import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import EssenceShelf from '../components/EssenceShelf';
import { api, type Essence } from '../lib/api';
import { rgbCss } from '../lib/color';

// Main Stage (spec §4.2.1): drag an essence bottle onto the target photo.
// The bottle empties, glowing threads cross the space, and the photo
// morphs continuously — no discrete before/after jump.
export default function MainStage() {
  const [essences, setEssences] = useState<Essence[]>([]);
  const [targetPath, setTargetPath] = useState<string | null>(null);
  const [baseSrc, setBaseSrc] = useState<string | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const shelfRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const particlesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    refreshEssences();
  }, []);

  // Dev-only automation hook — see appBridge.d.ts. Lets an agent/e2e run
  // bypass the native file dialog and real drag event, neither of which can
  // be driven from a headless test.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    window.__testHooks = {
      ...window.__testHooks,
      stage: {
        chooseTargetPath: (path: string) => chooseTarget(path),
        applyEssenceById: (essenceId: string) => applyEssence(essenceId, window.innerWidth / 2, window.innerHeight / 2),
        state: () => ({ targetPath, essenceCount: essences.length, isApplying, hasFinal: !!baseSrc }),
      },
    };
  }, [essences, targetPath, isApplying, baseSrc]);

  async function refreshEssences() {
    try {
      setEssences(await api.listEssences());
    } catch {
      // sidecar not up yet — shelf just stays empty until it is
    }
  }

  async function deleteEssence(target: Essence) {
    try {
      await api.deleteEssence(target.id);
      setEssences((prev) => prev.filter((e) => e.id !== target.id));
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Could not delete essence.');
      setTimeout(() => setStatus(null), 2000);
    }
  }

  async function chooseTarget(overridePath?: string) {
    const path = overridePath ?? (await window.appBridge.openImageDialog());
    if (!path) return;
    setTargetPath(path);
    setBaseSrc(await window.appBridge.readImageAsDataUrl(path));
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    const essenceId = e.dataTransfer.getData('text/essence-id');
    if (!essenceId) return;
    await applyEssence(essenceId, e.clientX, e.clientY);
  }

  async function applyEssence(essenceId: string, dropX: number, dropY: number) {
    if (!targetPath) {
      setStatus('Choose a target photo first.');
      setTimeout(() => setStatus(null), 1800);
      return;
    }
    if (isApplying) return;
    const essence = essences.find((x) => x.id === essenceId);
    if (!essence) return;

    setIsApplying(true);
    setStatus(`Distilling "${essence.name}" into the photo…`);
    playThreadAnimation(dropX, dropY, essence.color);

    try {
      const result = await api.applyEssence(essenceId, targetPath);
      await crossfadeSteps(result.steps);
      setBaseSrc(result.final);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Application failed.');
    } finally {
      setIsApplying(false);
      setTimeout(() => setStatus(null), 1500);
    }
  }

  // The bottle "tips and empties" toward the drop point as a scatter of
  // glowing particles traveling from the shelf. Purely cosmetic — runs
  // independent of (and typically finishes before) the actual apply call.
  function playThreadAnimation(dropX: number, dropY: number, color: [number, number, number]) {
    const layer = particlesRef.current;
    const shelfRect = shelfRef.current?.getBoundingClientRect();
    if (!layer || !shelfRect) return;
    const originX = shelfRect.left + shelfRect.width / 2;
    const originY = shelfRect.top + 48;

    const count = 8;
    for (let i = 0; i < count; i++) {
      const dot = document.createElement('div');
      dot.className = 'thread-particle';
      dot.style.background = rgbCss(color, 0.9);
      dot.style.boxShadow = `0 0 10px 2px ${rgbCss(color, 0.6)}`;
      layer.appendChild(dot);
      gsap.set(dot, { x: originX, y: originY + i * 5, opacity: 0 });
      gsap
        .timeline({ onComplete: () => dot.remove() })
        .to(dot, { opacity: 1, duration: 0.12 })
        .to(
          dot,
          {
            x: dropX + (Math.random() - 0.5) * 40,
            y: dropY + (Math.random() - 0.5) * 40,
            duration: 0.65 + Math.random() * 0.25,
            ease: 'power2.inOut',
          },
          0.04 * i,
        )
        .to(dot, { opacity: 0, duration: 0.3 }, '-=0.2');
    }
  }

  // Animation-generation decoupling (spec §4.2.1): the sidecar returns every
  // step already computed, but the crossfade below always runs over a fixed
  // wall-clock duration regardless of how long that took — the animation
  // clock and the generation clock are independent. When real diffusion
  // streams steps progressively, only the "wait for steps" side of this
  // changes; the fixed-duration blend here stays the same.
  function crossfadeSteps(steps: string[]): Promise<void> {
    return new Promise((resolve) => {
      const layer = overlayRef.current;
      if (!layer) return resolve();

      Promise.all(steps.map(preload)).then(() => {
        layer.innerHTML = '';
        const imgs = steps.map((src) => {
          const img = document.createElement('img');
          img.src = src;
          img.className = 'absolute inset-0 w-full h-full object-contain';
          img.style.opacity = '0';
          layer.appendChild(img);
          return img;
        });
        gsap.set(imgs[0], { opacity: 1 });

        const tl = gsap.timeline({
          onComplete: () => resolve(),
        });
        for (let i = 1; i < imgs.length; i++) {
          tl.to(imgs[i], { opacity: 1, duration: 0.3, ease: 'sine.inOut' }, i * 0.22);
        }
      });
    });
  }

  return (
    <div className="flex h-full">
      <div
        className="flex-1 flex items-center justify-center p-10 relative"
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        {!baseSrc && (
          <button
            onClick={() => chooseTarget()}
            className="border border-dashed border-white/15 rounded-card px-10 py-16 text-ink-soft hover:text-ink hover:border-gold/50 transition-colors font-body"
          >
            Click to choose a photo, or drop one here
          </button>
        )}

        {baseSrc && (
          <div className="relative w-full h-full max-w-3xl max-h-[70vh]">
            <img src={baseSrc} alt="Target" className="absolute inset-0 w-full h-full object-contain rounded-card shadow-2xl" />
            <div ref={overlayRef} className="absolute inset-0 rounded-card overflow-hidden pointer-events-none" />
            <button
              onClick={() => chooseTarget()}
              className="absolute -top-3 -right-3 bg-surface text-ink-soft hover:text-ink text-xs px-3 py-1 rounded-full border border-white/10"
            >
              Change photo
            </button>
          </div>
        )}

        {status && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-charcoal/90 text-ink-soft text-sm px-4 py-2 rounded-full border border-white/10">
            {status}
          </div>
        )}
      </div>

      <EssenceShelf ref={shelfRef} essences={essences} onDelete={deleteEssence} />

      {/* Fixed viewport-space layer for the thread/particle animation. */}
      <div ref={particlesRef} className="fixed inset-0 pointer-events-none z-40" />
    </div>
  );
}

function preload(src: string): Promise<void> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = () => resolve();
    img.src = src;
  });
}
