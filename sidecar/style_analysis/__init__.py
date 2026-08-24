"""Reference-image style analysis: palette, texture, stroke/directional-flow,
and general style statistics. Distillation-time only — see
essence.py's extract_essence for the integration point. Each analyzer takes
a PIL Image and returns one of the Pydantic profiles from essence_models.py.

All four run at a fixed, small analysis resolution (see _common.py),
independent of the essence's stored thumbnail or the generation pipeline's
working resolution — this is cheap CV/NumPy work, not something that
benefits from a bigger input.
"""
