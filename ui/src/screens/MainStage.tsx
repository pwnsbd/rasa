import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import EssenceShelf from '../components/EssenceShelf';
import { api, type ApplyMode, type BlendMode, type Essence } from '../lib/api';
import { rgbCss } from '../lib/color';
import { DEFAULT_INTENSITY, STEPS_HIGH_DETAIL, STEPS_STANDARD, intensityToParams, suggestIntensity } from '../lib/styleIntensity';

// Main Stage (spec §4.2.1): drag an essence bottle onto the target photo.
// The bottle empties, glowing threads cross the space, and the photo
// morphs continuously — no discrete before/after jump.
export default function MainStage() {
  const [essences, setEssences] = useState<Essence[]>([]);
  const [targetPath, setTargetPath] = useState<string | null>(null);
  const [baseSrc, setBaseSrc] = useState<string | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  // True only while waiting on the sidecar (not during the crossfade), so
  // the spinner clears the moment the result starts fading in.
  const [isWaiting, setIsWaiting] = useState(false);
  const [waitSeconds, setWaitSeconds] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  // Style intensity + quality controls (Main Stage artistic controls,
  // previously deferred). Left at their defaults, these reproduce exactly
  // what apply used to send with no controls at all — see
  // lib/styleIntensity.ts.
  const [intensity, setIntensity] = useState(DEFAULT_INTENSITY);
  // True once the user has dragged the intensity slider themselves. Until
  // then, applyEssence picks a per-essence suggested starting point (see
  // lib/styleIntensity.ts's suggestIntensity) instead of one flat default
  // for every essence — same "suggested, but overridable" pattern as
  // segmentation.py's suggest_subject_params. Any manual drag becomes a
  // fixed choice that applies to every essence from then on, matching what
  // the visible slider position promises.
  const [intensityTouched, setIntensityTouched] = useState(false);
  const [highDetail, setHighDetail] = useState(false);
  // 0..1, default 1.0 (full restore): IP-Adapter's embedding carries the
  // essence's own color along with its texture, which can otherwise tint
  // the whole photo toward the essence's hue (reported directly against a
  // real run — see sidecar/color_transfer.py). 1.0 restores the photo's
  // original color fully; 0.0 lets the essence's color through untouched,
  // same as the pipeline's original behavior; values between are a real
  // blend instead of an on/off toggle.
  const [colorPreservation, setColorPreservation] = useState(1.0);
  // 0..1, default 0.0 (off): overlays a faint directional grain oriented to
  // the essence's own measured stroke pattern (see sidecar/stroke_texture.py)
  // — the first control that reacts to which *specific* essence was
  // dropped rather than applying the same adjustment to every essence
  // alike. New and not yet validated against a real photo the way
  // colorPreservation was, so it stays off until a user opts in; a silent
  // no-op for essences with no detected stroke field (blended essences,
  // ones saved before this existed, or failed stroke analysis).
  const [strokeAmount, setStrokeAmount] = useState(0);
  // Subject (rembg+face two-pass, already validated on portraits) stays the
  // default. Depth (continuous depth-driven two-pass — see
  // sidecar/depth.py) is the new option for photos without one clear
  // subject: landscapes, group shots, product shots — the kind of thing a
  // flat filter has no way to react to at all.
  const [blendMode, setBlendMode] = useState<BlendMode>('subject');
  // "restyle" (default): the SDXL diffusion pipeline above -- everything
  // this screen did before. "texture_overlay": a second, non-diffusion
  // engine (see sidecar/texture_overlay.py) that composites the essence's
  // own material swatch onto the target via classical bump/emboss shading
  // instead of regenerating the photo -- the right tool for an essence
  // that's really a material (glass, canvas, paper grain) rather than a
  // painterly style, where diffusion only drifts the target's content away
  // from what it actually was. Every restyle control above is ignored by
  // the sidecar in this mode (see generation.apply_essence's docstring);
  // errors (e.g. dropping it on an essence with no saved material swatch)
  // surface through the same status toast as any other apply failure.
  const [mode, setMode] = useState<ApplyMode>('restyle');
  const [textureOverlayAmount, setTextureOverlayAmount] = useState(1.0);

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
        state: () => ({
          targetPath,
          essenceCount: essences.length,
          isApplying,
          hasFinal: !!baseSrc,
          intensity,
          intensityTouched,
          highDetail,
          colorPreservation,
          strokeAmount,
          blendMode,
          mode,
          textureOverlayAmount,
        }),
        setIntensity: (value: number) => {
          setIntensityTouched(true);
          setIntensity(value);
        },
        setHighDetail: (value: boolean) => setHighDetail(value),
        setColorPreservation: (value: number) => setColorPreservation(value),
        setStrokeAmount: (value: number) => setStrokeAmount(value),
        setBlendMode: (value: BlendMode) => setBlendMode(value),
        setMode: (value: ApplyMode) => setMode(value),
        setTextureOverlayAmount: (value: number) => setTextureOverlayAmount(value),
      },
    };
  }, [
    essences,
    targetPath,
    isApplying,
    baseSrc,
    intensity,
    intensityTouched,
    highDetail,
    colorPreservation,
    strokeAmount,
    blendMode,
    mode,
    textureOverlayAmount,
  ]);

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
    // crossfadeSteps leaves its last (fully-opaque) step image sitting in
    // this layer once the animation finishes — it only clears at the start
    // of the *next* crossfade. Without clearing it here too, picking a new
    // photo left the previous result stacked visually on top of it. (Bug
    // reported directly: "change photo does not remove the old photo, they
    // stack.")
    if (overlayRef.current) overlayRef.current.innerHTML = '';
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
    setIsWaiting(true);
    setStatus(`Distilling "${essence.name}" into the photo…`);
    playThreadAnimation(dropX, dropY, essence.color);

    try {
      // Until the user drags the slider themselves, each essence gets its
      // own suggested starting intensity (see lib/styleIntensity.ts) rather
      // than one flat default applied identically to every essence — and
      // the slider itself moves to show what's actually being used.
      const effectiveIntensity = intensityTouched ? intensity : suggestIntensity(essence);
      if (!intensityTouched) setIntensity(effectiveIntensity);
      const { strength, controlnetScale } = intensityToParams(effectiveIntensity);
      const result = await api.applyEssence(essenceId, targetPath, {
        strength,
        controlnetScale,
        steps: highDetail ? STEPS_HIGH_DETAIL : STEPS_STANDARD,
        colorPreservation,
        strokeAmount,
        blendMode,
        mode,
        textureOverlayAmount,
      });
      setIsWaiting(false);
      await crossfadeSteps(result.steps);
      setBaseSrc(result.final);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Application failed.');
    } finally {
      setIsWaiting(false);
      setIsApplying(false);
      setTimeout(() => setStatus(null), 1500);
    }
  }

  useEffect(() => {
    if (!isWaiting) return;
    setWaitSeconds(0);
    const started = Date.now();
    const timer = setInterval(() => setWaitSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [isWaiting]);

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
            {isWaiting && (
              <div
                role="status"
                aria-live="polite"
                className="absolute inset-0 rounded-card bg-charcoal/55 backdrop-blur-[2px] flex flex-col items-center justify-center gap-3 pointer-events-none"
              >
                <div className="w-10 h-10 rounded-full border-2 border-white/15 border-t-gold animate-spin" />
                <span className="text-ink text-sm font-body">Restyling… {waitSeconds}s</span>
              </div>
            )}
            <button
              onClick={() => chooseTarget()}
              className="absolute -top-3 -right-3 bg-surface text-ink-soft hover:text-ink text-xs px-3 py-1 rounded-full border border-white/10"
            >
              Change photo
            </button>

            {/* Main Stage artistic controls: a single intensity slider drives
                strength + controlnet_scale together (see lib/styleIntensity.ts)
                rather than exposing those two raw, easy-to-misuse knobs directly. */}
            <div className="absolute -bottom-14 left-1/2 -translate-x-1/2 flex items-center flex-wrap justify-center gap-4 bg-charcoal/90 text-ink-soft text-xs px-4 py-2 rounded-card border border-white/10 max-w-[90vw]">
              {/* Engine switch: two entirely different apply paths, not a
                  blend of one (see sidecar/texture_overlay.py and
                  generation.apply_essence's own docstring for why a
                  material-like essence needs a non-diffusion engine
                  instead of another restyle slider). Everything below
                  swaps based on which is selected. */}
              <div className="flex items-center gap-1 border border-white/10 rounded-full p-0.5" role="group" aria-label="Apply engine">
                {(
                  [
                    { m: 'restyle' as const, label: 'Restyle', title: 'The diffusion pipeline — reinterprets the photo in the essence\'s style' },
                    { m: 'texture_overlay' as const, label: 'Overlay', title: 'Classical bump/emboss compositing — prints the essence\'s own material (glass, canvas, paper) onto the photo unchanged, no diffusion' },
                  ]
                ).map(({ m, label, title }) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    title={title}
                    className={`px-2 py-0.5 rounded-full transition-colors ${
                      mode === m ? 'bg-gold/20 text-gold' : 'text-ink-soft hover:text-ink'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {mode === 'restyle' ? (
                <>
                  <label className="flex items-center gap-2" title={intensityTouched ? undefined : "Starting point suggested per essence — drag to set your own for every essence"}>
                    <span>Subtle</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={intensity}
                      onChange={(e) => {
                        setIntensityTouched(true);
                        setIntensity(parseFloat(e.target.value));
                      }}
                      className="w-28 accent-gold"
                      aria-label="Style intensity"
                    />
                    <span>Strong</span>
                  </label>
                  <button
                    onClick={() => setHighDetail((v) => !v)}
                    className={`px-2 py-0.5 rounded-full border transition-colors ${
                      highDetail ? 'border-gold/60 text-gold' : 'border-white/10 text-ink-soft hover:text-ink'
                    }`}
                    title="More denoising steps: crisper detail, takes longer"
                  >
                    {highDetail ? 'High detail' : 'Standard'}
                  </button>
                  <label className="flex items-center gap-2" title="How much of the photo's own color to keep — low lets the essence's color through">
                    <span>Essence color</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={colorPreservation}
                      onChange={(e) => setColorPreservation(parseFloat(e.target.value))}
                      className="w-20 accent-gold"
                      aria-label="Color preservation"
                    />
                    <span>Original color</span>
                  </label>
                  <label
                    className="flex items-center gap-2"
                    title="Faint directional grain matching the dropped essence's own brush/stroke pattern — no effect for essences without a detected stroke pattern"
                  >
                    <span>Stroke grain</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={strokeAmount}
                      onChange={(e) => setStrokeAmount(parseFloat(e.target.value))}
                      className="w-16 accent-gold"
                      aria-label="Stroke texture amount"
                    />
                  </label>
                  <div className="flex items-center gap-1 border border-white/10 rounded-full p-0.5" role="group" aria-label="Blend mode">
                    {(
                      [
                        { bm: 'subject' as const, label: 'Subject', title: 'Best for portraits — preserves the detected subject/face' },
                        { bm: 'depth' as const, label: 'Depth', title: 'Best for landscapes/products — foreground stays crisp, background stylizes more with distance' },
                        { bm: 'none' as const, label: 'Off', title: 'One flat pass over the whole photo' },
                      ]
                    ).map(({ bm, label, title }) => (
                      <button
                        key={bm}
                        onClick={() => setBlendMode(bm)}
                        title={title}
                        className={`px-2 py-0.5 rounded-full transition-colors ${
                          blendMode === bm ? 'bg-gold/20 text-gold' : 'text-ink-soft hover:text-ink'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <label className="flex items-center gap-2" title="How strongly the essence's material relief reads on the photo">
                  <span>Subtle</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={textureOverlayAmount}
                    onChange={(e) => setTextureOverlayAmount(parseFloat(e.target.value))}
                    className="w-28 accent-gold"
                    aria-label="Overlay strength"
                  />
                  <span>Strong</span>
                </label>
              )}
            </div>
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
