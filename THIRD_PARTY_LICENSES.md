# Third-party models and licenses

Rasa's own source code is MIT-licensed (see [LICENSE](LICENSE)). At runtime it
downloads and runs several third-party machine learning models, each under
its own license. Verified before shipping — not assumed — via each model's
official Hugging Face model card as of 2026-08. If any of these models are
swapped or upgraded, re-check the new checkpoint's license before shipping;
the license can differ between variants of the "same" model (see
Depth Anything V2 below for exactly that trap).

| Model | Used for | License | Commercial use | Source |
|---|---|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | Base image generation | CreativeML Open RAIL++-M | Allowed — no revenue cap, no royalties. Carries *use-based* restrictions (can't be used to generate certain categories of harmful/illegal content) rather than business restrictions. | [Model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| `xinsir/controlnet-tile-sdxl-1.0` | Tile ControlNet (structure preservation) | CreativeML Open RAIL++-M Addendum | Same family/terms as SDXL above. | [Model card](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0) |
| `h94/IP-Adapter` | InstantStyle image-prompt adapter | Apache 2.0 | Fully permissive. | [Model card](https://huggingface.co/h94/IP-Adapter) |
| `madebyollin/sdxl-vae-fp16-fix` | VAE (fp16 stability fix) | MIT | Fully permissive. | [Model card](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| `depth-anything/Depth-Anything-V2-Small-hf` | Depth estimation (depth blend mode, Media Page parallax) | Apache 2.0 | Fully permissive — **but only the Small checkpoint**. The Base/Large/Giant Depth Anything V2 variants are CC-BY-NC-4.0 (commercial use forbidden outright). Rasa is pinned to Small specifically (`sidecar/depth.py`) — do not upgrade to a larger variant without re-checking this. | [Model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| `rembg` (u2net model) | Subject/background segmentation | Apache-2.0-derived (MIT wrapper over an Apache-licensed model) | Fully permissive. Note: `rembg` supports other backend models (e.g. RMBG-2.0) that require a **paid** commercial license — Rasa does not use those; `segmentation.py` uses the original u2net model specifically. | [rembg repo](https://github.com/danielgatis/rembg) |

## Open-source libraries

`torch`, `diffusers`, `transformers`, `safetensors`, `opencv-python-headless`,
`pillow`/`pillow-heif`, `numpy`, `scipy`, `scikit-image`, `fastapi`,
`uvicorn`, `pydantic`, `three.js`, `react`, `gsap`, and the rest of
`sidecar/requirements.txt` / `ui/package.json` are all under standard
permissive licenses (Apache 2.0, BSD, or MIT). Full per-package license
enumeration is a good candidate for automated tooling (`pip-licenses` for
the sidecar, `license-checker` for the UI) before a real release — not
manually re-verified line-by-line here.

## Attribution requirement

Apache 2.0 (IP-Adapter, Depth Anything V2 Small, rembg's underlying model)
requires the license text and any NOTICE be preserved with redistributed
copies. This file, alongside each project's own repository, satisfies that
for Rasa's distribution.

## Not legal advice

This document is a good-faith summary compiled by reading each model's
published license, not a legal opinion. Get an actual license review before
a real commercial release — particularly around the RAIL++-M license's
use-based restrictions, which should also be reflected in Rasa's own
acceptable-use terms for end users (see the acceptable-use note this implies
for any future Terms of Service).
