# Rasa — Product & Technical Specification

**Rasa** (Sanskrit: essence, juice, flavor — also the emotional essence evoked by a work of art) is a local-first, GPU-aware desktop application for extracting the visual style of a reference image and reapplying it to other photos. It is built around a swappable pipeline architecture and an alchemy-inspired interface.

---

## 1. Product Philosophy

- **No subscriptions.** Fully local, one-time setup, no accounts, no cloud dependency.
- **GPU-aware with CPU fallback.** Detects available hardware at launch (CUDA compute capability, VRAM) and adapts; primary dev target is an RTX 5070 (CUDA 12.6) on Windows, but the detection layer is written generically, not hard-coded to one GPU.
- **Minimal UI, single clear pipeline.** Each screen does exactly one job. No overlapping responsibilities.
- **Swappable internals.** Base generation model and style-extraction technique are both interchangeable behind clean interfaces, so the app isn't married to any one model as the open-weights landscape moves.

---

## 2. Core Pipeline

### 2.1 Concept

Two distinct actions, kept deliberately separate:

1. **Extraction** — take a reference image → distill a reusable **Essence** (a style embedding + metadata), stored locally.
2. **Application** — take a target photo + a saved Essence → generate a new image where the target's content is preserved but its style is transformed to match the Essence.

### 2.2 Model Architecture (swappable by design)

Three independently swappable layers, each behind an interface:

**a) Base generation model interface**
- `BaseModel.generate(content_image, style_conditioning) -> image`
- Initial implementation: **Flux** (currently top of the open-weights leaderboards for both text-to-image and editing quality — see Artificial Analysis open-weights leaderboard).
- Designed so a stronger open-weights base model can be swapped in later (e.g. if a new model tops the leaderboard) without touching the extraction or UI layers.

**b) Style-extraction technique interface**
- `StyleExtractor.extract(reference_image) -> embedding`
- `StyleExtractor.apply(embedding, target_image) -> conditioning`
- Initial implementations:
  - **InstantStyle** (built on IP-Adapter, isolates style-carrying SDXL/Flux attention blocks from layout-carrying ones — cleaner separation of style from content, good default for general/photographic references).
  - **StyleShot** (stronger for heavy-texture/painterly references like brushstrokes; can over-emphasize texture on some references).
- User (or the app, based on heuristics) selects extraction technique per style, not globally — a painterly reference might use StyleShot, a photographic reference might use InstantStyle.

**c) Embedding storage layer**
- Each saved Essence = an embedding tensor + metadata (source thumbnail, extraction technique used, block/layer configuration, creation date).
- Stored as a local file/folder per Essence — small footprint since it's an embedding, not a copy of the base model or a checkpoint.

### 2.3 GPU/Device Detection

- On launch, detect CUDA availability and compute capability.
- If no compatible GPU is found, fall back to CPU (with an expected-slowness warning in UI).
- Dev/test target: RTX 5070, CUDA 12.6. Detection logic must not assume this specific card.

---

## 3. Provenance & Sharing System

### 3.1 Goal

Every image exported from Rasa should be traceable back to the Essence and technique used to create it — enabling a future "style marketplace" where users can download someone else's exported image and instantly reapply its exact style, without re-running extraction.

### 3.2 Mechanism

- On export, embed metadata invisibly in the file (PNG/JPEG metadata chunks): which Essence, which extraction technique, which base model version.
- This metadata must not interfere with normal image viewing/posting (Instagram, etc.) — the image displays identically everywhere.
- **Known limitation:** most public platforms (Instagram, Facebook, X, TikTok, Snapchat, LinkedIn, Reddit) strip custom metadata on upload because they re-encode images. Metadata reliably survives on direct file transfer (email, cloud storage, Discord, Slack) and, guaranteed, within Rasa's own ecosystem.
- **Mitigation / fallback:** Rasa hosts its own gallery/web page as the canonical sharing surface, where embedded metadata is always preserved and readable by the app.
- **Belt-and-suspenders option:** an optional subtle visible watermark alongside the hidden metadata, so even after a platform strips the data, viewers have a visual cue to find the original on Rasa's gallery.

### 3.3 Future: Style Marketplace (not v1, but designed for)

- A store/gallery where users upload Rasa-generated images.
- Other users can download those images and, because the provenance metadata is intact, immediately reuse the exact style as their own Essence — no re-extraction, no inference cost.

---

## 4. UI/UX Concept

### 4.1 Visual Direction

- **Alchemy workbench**, not a clinical editor. Two images "meeting" is treated as a ritual, not a filter application.
- **Tone:** calm dusk, not midnight — warm charcoal / deep plum backgrounds rather than pure black. Should feel magical and inviting, explicitly *not* dark/uncanny.
- **Format:** 2.5D (subtle depth/parallax), not full 3D — consistent with the DepthFrame project's approach. Avoids the complexity of a navigable 3D scene, which isn't needed since the core interaction is fundamentally two flat images meeting.
- **Iconography:** thin-line "alkaline"-style icons throughout; alchemical symbols specifically for logos and settings (flasks, distillation glyphs) rather than generic gear/folder icons.

### 4.2 The Four Zones (each a single-purpose screen/area)

1. **Main Stage** (application)
   - Large, centered target photo.
   - Essence shelf on the right: vertical stack of thumbnails (source-image recognizability first), each paired with a small bottle icon showing the essence's "color" as a secondary badge — not the primary visual, to avoid the problem of abstract swirls being unreadable on their own.
   - Interaction: drag an essence bottle onto the target photo.
   - Animation: the bottle tips and empties; the essence travels as glowing threads/smoke across the space into the photo; the target photo itself visibly, continuously morphs to take on the style — no discrete "before/after" jump.
   - **Animation-generation decoupling:** since diffusion happens in discrete, uneven steps, the animation clock must be decoupled from the generation clock. Interpolate/blend between completed steps so the visual reads as one smooth continuous transformation, even while the underlying generation is chunky.

2. **Distillation Room** (extraction)
   - Separate, focused view/window from the main stage — entering it doesn't dump the resulting essence back onto the shelf automatically; the user stays in this room to process multiple references back-to-back if desired.
   - Drop in a reference photo — it shimmers and distills — a glowing, lava-like liquid (colored/textured based on the actual style extracted) pours into an empty bottle outline until sealed.
   - Only when the user chooses to leave does the new Essence appear on the Main Stage shelf.

3. **Media Page** (gallery/archive)
   - Every finished creation is automatically saved here — nothing is lost even if not exported immediately.
   - Export action from here triggers the provenance-metadata embedding (Section 3) and optional watermark, then saves as a normal shareable file.

4. **Settings**
   - Model configuration: base model selection, style-extraction technique defaults, GPU/device status.
   - Alkaline-style thin-line icons; alchemical iconography consistent with the rest of the app.

### 4.3 Interaction Summary

| Action | Where | Visual |
|---|---|---|
| Extract a style | Distillation Room | Reference photo shimmers → lava-like liquid pours into a new bottle |
| Apply a style | Main Stage | Bottle empties → threads/smoke cross to target → target morphs continuously |
| Browse saved styles | Main Stage shelf | Thumbnail stack (recognizable) + bottle badge (mood/color) |
| View/export past work | Media Page | Gallery of finished creations; export embeds provenance metadata |
| Configure models/hardware | Settings | Alchemical icon set, GPU status, model swap controls |

---

## 5. Naming

- **App name: Rasa** (रस) — Sanskrit for essence/juice/flavor, and in Indian aesthetic theory, the emotional essence a work of art evokes in its audience. Fits both the literal function (style embeddings) and the intended feeling (calm, ritual-like creative act).
- Consistent with prior Sanskrit naming convention (Nidhi, Jal, Udgam, Nakshatra, Drishyasmriti).
- Saved style embeddings are called **Essences**.

---

## 6. Suggested Stack (consistent with existing tooling)

- **Frontend:** React + Vite + TypeScript, React Three Fiber for 2.5D depth/parallax (same pattern as DepthFrame), GSAP for animation sequencing (thread/smoke/pour effects, and interpolated diffusion-step blending).
- **Backend/inference:** Python, local diffusion pipeline (Flux base + InstantStyle/StyleShot adapters), device-agnostic model loading (CUDA detection → CPU fallback).
- **Storage:** Local filesystem for Essences (embedding + metadata) and Media Page archive; no cloud/account layer for v1.

---

## 7. Open Items / Not Yet Decided

- Exact file/folder schema for a saved Essence (embedding format, metadata fields).
- Heuristic (if any) for auto-suggesting InstantStyle vs. StyleShot based on reference image characteristics, vs. leaving it fully manual.
- Scope and timeline for the future style marketplace (explicitly deferred beyond v1).
- Exact metadata embedding format/spec (PNG chunk type, JPEG segment, etc.) and watermark visual design.
