# Contract — restyle performance (memory plan, cached prompts, progress)

**Purpose:** Make a restyle use the least memory the machine needs and pick the fastest placement that fits any GPU, and show live step progress with a slow-GPU warning.
**Files:** `sidecar/pipeline_manager.py`, `sidecar/generation.py`, `sidecar/essence_store.py` (encoder placement only), `sidecar/app.py`, `ui/src/lib/api.ts`, `ui/src/screens/MainStage.tsx`, new `sidecar/tests/test_memory_plan.py`, README, `docs/contracts/generation.md`.

## Why (measured 2026-10-04, RTX 5070 Ti Laptop 12 GB, 31 GB RAM)
- Single pass, 45 steps: 64 s. Default Subject blend (two passes): 125 s. Peak VRAM 11.7/12.2 GB; free RAM fell 17.5 → 2.6 GB. Under that pressure Windows spills/pages and runs take 6–10 min.
- `enable_model_cpu_offload` moves about 14 GB between RAM and GPU on every run.
- The text prompt is always `""` and the negative prompt is a constant, but both text encoders (about 1.6 GB) run on every apply.
- The CLIP-H image encoder (about 1.2 GB) is only needed at essence extraction; apply already uses stored `ip_adapter_image_embeds`.

## Inputs
- No new required `/apply` fields. `blend_mode` default changes `"subject"` → `"none"` in **both** `ApplyRequest` and the MainStage initial state.

## Behaviour
1. **Cached prompt embeddings.** After load, call `pipe.encode_prompt` once for `prompt=""` + `NEGATIVE_PROMPT` and cache `prompt_embeds`, `negative_prompt_embeds`, `pooled_prompt_embeds` and `negative_pooled_prompt_embeds`. `_run_generation` passes these instead of `prompt`/`negative_prompt`. The output must stay identical for the same seed.
2. **Memory plan**, chosen once at load by a **pure function** `choose_memory_plan(free_vram_bytes, total_vram_bytes, device) -> "resident" | "offload" | "cpu"`:
   - `resident`: UNet, ControlNet and VAE go on cuda. Text encoders and image encoder stay on **CPU**. Choose this when free VRAM ≥ resident need + safety margin. Use about 9.0 GB as the threshold, as a named constant.
   - `offload`: today's `enable_model_cpu_offload` path, unchanged.
   - `cpu`: no CUDA available.
   - Expose the plan in `pipeline_manager.status()` and `/health` → `model.memory_plan`.
3. **Device handling.** In resident mode, `pipe._execution_device` may report CPU because components are mixed. Add `pipeline_manager.compute_device()` and use it everywhere `_execution_device` is used today (`generation.py`, `essence_store.py`).
4. **Essence extraction in resident mode.** Move `image_encoder` to cuda only for the duration of embedding (use a context manager), then back to CPU. Extraction results must not change.
5. **Progress.** A thread-safe module-level tracker updated from `callback_on_step_end`: `{active, pass_index, pass_count, step, total_steps, sec_per_step (rolling median of the last 3), slow}`. `slow = sec_per_step > SLOW_STEP_S` (6.0 s). Expose it at `GET /apply/progress`. Reset at apply start; mark inactive at the end or on error.
6. **UI.** While `isWaiting`, MainStage polls `api.applyProgress()` every 1 s through the existing `sidecar:call` proxy, never `fetch`. The spinner label becomes `Restyling… step 12/27 · 1.8s/step` (show `pass 2/2` when there are two passes). When `slow`, show a second line: "Your GPU memory is full — close other GPU-heavy apps (browsers, games, wallpaper engines) for a much faster restyle." Before the first step arrives, keep the existing `Restyling… Ns`.

## Errors
- If moving to cuda for resident mode raises OOM, fall back to `offload` and log it. It must never crash the load.
- A progress poll while idle returns `{active:false}`, not an error.

## Perf budget
On the dev laptop with a single pass and 45 steps: under 50 s and peak VRAM under 10 GB. Pawan's machine measures this after merge, not the builder.

## Out of scope
Changing step counts, the scheduler or the models. Switching plans mid-session. Texture-overlay mode. Visual-quality changes.

## Done-check
- `sidecar/venv/Scripts/python -m pytest sidecar/tests -q` passes (main checkout venv, absolute path). New tests cover:
  - `choose_memory_plan` thresholds
  - the progress tracker (rolling median, `slow` flag, reset)
  - `_run_generation` passing cached embeds, not prompt strings (stub pipe)
  - the `/apply/progress` idle response
- `npm run build:ui` passes.
- Conductor benchmarks on the GPU after merge, and Pawan confirms the look is unchanged.
