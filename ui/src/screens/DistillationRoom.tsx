import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import DistillationBottle from '../components/DistillationBottle';
import CauldronVessel from '../components/CauldronVessel';
import { BottleBadge } from '../components/icons';
import { api, type BlendIngredientInput, type Essence } from '../lib/api';
import { estimateAverageColor, rgbCss } from '../lib/color';

type Phase = 'idle' | 'shimmering' | 'pouring' | 'sealed';
type Mode = 'single' | 'cauldron';

interface Ingredient {
  id: string;
  type: 'image' | 'essence';
  label: string;
  thumbnailUrl: string;
  color: [number, number, number];
  weight: number;
  imagePath?: string;
  essenceId?: string;
}

function nextIngredientId(): string {
  return `ing-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function blendPreviewColor(ingredients: Ingredient[]): [number, number, number] {
  const total = ingredients.reduce((s, i) => s + i.weight, 0);
  if (total <= 0) return [201, 163, 92]; // gold — same idle fallback DistillationBottle already used
  const sum = ingredients.reduce<[number, number, number]>(
    (acc, i) => [acc[0] + i.color[0] * i.weight, acc[1] + i.color[1] * i.weight, acc[2] + i.color[2] * i.weight],
    [0, 0, 0],
  );
  return [Math.round(sum[0] / total), Math.round(sum[1] / total), Math.round(sum[2] / total)];
}

// Distillation Room (spec §4.2.2): a focused space separate from the Main
// Stage. Extracting doesn't dump the essence onto the shelf automatically —
// it only appears there once the user navigates away, which App.tsx gets
// for free by only mounting the active zone (MainStage refetches on mount).
//
// Two modes sharing one pour/seal ritual: Single (one reference photo, the
// original flow) and Cauldron (blend existing Essences and/or several
// reference photos into one new Essence — same underlying operation either
// way, see sidecar/essence_store.py's blend_essences). Both end at the same
// DistillationBottle fill animation and the same "sealed this session"
// tray — a blended Essence is a completely normal Essence afterward.
export default function DistillationRoom() {
  const [mode, setMode] = useState<Mode>('single');

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [fill, setFill] = useState(0);
  const [pendingEssence, setPendingEssence] = useState<Essence | null>(null);
  const [sessionEssences, setSessionEssences] = useState<Essence[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [ingredients, setIngredients] = useState<Ingredient[]>([]);
  const [availableEssences, setAvailableEssences] = useState<Essence[]>([]);
  const [isBrewing, setIsBrewing] = useState(false);

  const fillState = useRef({ v: 0 });

  useEffect(() => {
    if (mode !== 'cauldron') return;
    api.listEssences().then(setAvailableEssences).catch(() => {});
  }, [mode]);

  // Dev-only automation hook — see appBridge.d.ts / MainStage.tsx.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    window.__testHooks = {
      ...window.__testHooks,
      distillation: {
        chooseReferencePath: (path: string) => chooseReference(path),
        seal: () => seal(),
        setMode: (m: Mode) => setMode(m),
        addImageIngredient: (path: string) => addImageIngredient(path),
        addEssenceIngredient: (essenceId: string) => addEssenceIngredientById(essenceId),
        distillBlend: () => distillBlend(),
        state: () => ({
          mode,
          phase,
          sessionCount: sessionEssences.length,
          pendingName: pendingEssence?.name,
          ingredientCount: ingredients.length,
        }),
      },
    };
  }, [mode, phase, sessionEssences, pendingEssence, ingredients]);

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
    setIngredients([]); // no-op for Single mode; clears the Cauldron's tray
    setFill(0);
    setPhase('idle');
  }

  async function addImageIngredient(path: string) {
    const dataUrl = await window.appBridge.readImageAsDataUrl(path);
    const color = await estimateAverageColor(dataUrl);
    setIngredients((prev) => [
      ...prev,
      {
        id: nextIngredientId(),
        type: 'image',
        label: path.split(/[\\/]/).pop() ?? path,
        thumbnailUrl: dataUrl,
        color,
        weight: 1,
        imagePath: path,
      },
    ]);
  }

  async function addPhotos() {
    const paths = await window.appBridge.openImagesDialog();
    for (const path of paths) {
      await addImageIngredient(path);
    }
  }

  function addEssenceIngredientById(essenceId: string) {
    const essence = availableEssences.find((e) => e.id === essenceId);
    if (!essence || ingredients.some((i) => i.essenceId === essenceId)) return;
    setIngredients((prev) => [
      ...prev,
      {
        id: nextIngredientId(),
        type: 'essence',
        label: essence.name,
        thumbnailUrl: essence.thumbnail,
        color: essence.color,
        weight: 1,
        essenceId: essence.id,
      },
    ]);
  }

  function onVesselDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  function onVesselDrop(e: React.DragEvent) {
    e.preventDefault();
    const essenceId = e.dataTransfer.getData('text/essence-id');
    if (essenceId) addEssenceIngredientById(essenceId);
  }

  function removeIngredient(id: string) {
    setIngredients((prev) => prev.filter((i) => i.id !== id));
  }

  function updateWeight(id: string, weight: number) {
    setIngredients((prev) => prev.map((i) => (i.id === id ? { ...i, weight } : i)));
  }

  async function distillBlend() {
    if (ingredients.length === 0 || isBrewing) return;
    setError(null);
    setIsBrewing(true);
    try {
      const payload: BlendIngredientInput[] = ingredients.map((i) =>
        i.type === 'image'
          ? { type: 'image', imagePath: i.imagePath!, weight: i.weight }
          : { type: 'essence', essenceId: i.essenceId!, weight: i.weight },
      );
      const result = await api.blendEssences(payload);
      setPendingEssence(result);
      setPhase('pouring');
      pourIn();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Blending failed.');
    } finally {
      setIsBrewing(false);
    }
  }

  const totalWeight = ingredients.reduce((s, i) => s + i.weight, 0);
  const mixColor = blendPreviewColor(ingredients);
  const mixFill = ingredients.length === 0 ? 0.1 : Math.min(1, 0.35 + ingredients.length * 0.2);

  return (
    <div className="h-full flex flex-col items-center justify-center gap-8 p-10">
      <h1 className="font-display text-2xl text-ink">Distillation Room</h1>

      {phase === 'idle' && (
        <div className="flex items-center gap-1 border border-white/10 rounded-full p-0.5 text-xs" role="group" aria-label="Distillation mode">
          {(['single', 'cauldron'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1 rounded-full transition-colors ${
                mode === m ? 'bg-gold/20 text-gold' : 'text-ink-soft hover:text-ink'
              }`}
            >
              {m === 'single' ? 'Single' : 'Cauldron'}
            </button>
          ))}
        </div>
      )}

      {phase === 'idle' && mode === 'single' && (
        <button
          onClick={() => chooseReference()}
          className="border border-dashed border-white/15 rounded-card px-10 py-16 text-ink-soft hover:text-ink hover:border-gold/50 transition-colors font-body"
        >
          Drop in a reference photo to distill its Essence
        </button>
      )}

      {phase === 'idle' && mode === 'cauldron' && (
        <div className="flex flex-col items-center gap-5 w-full max-w-2xl">
          <p className="text-ink-soft text-xs text-center max-w-md">
            Mix existing Essences and/or several reference photos into one new Essence — a blend, not a preset.
          </p>

          <div onDragOver={onVesselDragOver} onDrop={onVesselDrop} title="Drag an essence bottle here to add it">
            <CauldronVessel fill={mixFill} color={mixColor} swirling={ingredients.length > 0} />
          </div>

          {ingredients.length > 0 && (
            <div className="flex flex-wrap justify-center gap-4 max-w-xl">
              {ingredients.map((ing) => {
                const share = totalWeight > 0 ? ing.weight / totalWeight : 0;
                const dim = 40 + Math.sqrt(Math.max(share, 0)) * 40;
                return (
                  <div key={ing.id} className="flex flex-col items-center gap-1 w-20">
                    <div className="relative" style={{ width: dim, height: dim }}>
                      <img
                        src={ing.thumbnailUrl}
                        alt=""
                        style={{ width: dim, height: dim }}
                        className="rounded-full object-cover border border-white/10"
                      />
                      <button
                        onClick={() => removeIngredient(ing.id)}
                        title={`Remove ${ing.label}`}
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-charcoal border border-white/10 text-ink-soft hover:text-red-300 flex items-center justify-center text-[10px] leading-none"
                      >
                        ×
                      </button>
                    </div>
                    <p className="text-ink-soft text-[10px] truncate w-full text-center" title={ing.label}>
                      {ing.label}
                    </p>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={ing.weight}
                      onChange={(e) => updateWeight(ing.id, parseFloat(e.target.value))}
                      className="w-16 accent-gold"
                      aria-label={`${ing.label} weight`}
                    />
                    <span className="text-ink-soft text-[10px]">{Math.round(share * 100)}%</span>
                  </div>
                );
              })}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              onClick={addPhotos}
              className="border border-dashed border-white/15 rounded-full px-4 py-1.5 text-xs text-ink-soft hover:text-ink hover:border-gold/50 transition-colors"
            >
              + Add photo(s)
            </button>
            {ingredients.length > 0 && (
              <button
                onClick={distillBlend}
                disabled={isBrewing}
                className="bg-gold/90 hover:bg-gold disabled:opacity-50 text-charcoal text-xs px-4 py-1.5 rounded-full font-medium"
              >
                {isBrewing ? 'Distilling…' : 'Distill blend'}
              </button>
            )}
          </div>

          {availableEssences.length > 0 && (
            <div className="w-full">
              <p className="text-ink-soft text-[10px] mb-1.5 text-center">Existing Essences — drag one into the cauldron</p>
              <div className="flex gap-2 overflow-x-auto justify-center pb-1">
                {availableEssences.map((e) => (
                  <img
                    key={e.id}
                    src={e.thumbnail}
                    alt={e.name}
                    title={e.name}
                    draggable
                    onDragStart={(ev) => {
                      ev.dataTransfer.setData('text/essence-id', e.id);
                      ev.dataTransfer.effectAllowed = 'copy';
                    }}
                    className="shrink-0 w-9 h-9 rounded object-cover border border-white/10 cursor-grab active:cursor-grabbing"
                    style={{ borderColor: rgbCss(e.color, 0.5) }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {phase !== 'idle' && (
        <div className="flex items-center gap-12">
          <div className="relative w-56 h-56 rounded-card overflow-hidden border border-white/10 flex items-center justify-center bg-charcoal/40">
            {previewUrl && <img src={previewUrl} alt="Reference" className="w-full h-full object-cover" />}
            {!previewUrl && ingredients.length > 0 && (
              <div className="grid grid-cols-2 gap-1 p-2 w-full h-full">
                {ingredients.slice(0, 4).map((ing) => (
                  <img key={ing.id} src={ing.thumbnailUrl} alt="" className="w-full h-full object-cover rounded" />
                ))}
              </div>
            )}
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
