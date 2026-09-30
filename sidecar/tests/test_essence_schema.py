"""Essence schema round-trip and backward compatibility (spec Phase 20).
Exercises essence_models.EssenceMeta directly, plus essence.list_essences()
against a temp essences_dir — list_essences() doesn't touch the SDXL
pipeline (only extract_essence does), so this needs no GPU.
"""
import json

import pytest

import essence_store
import paths
from essence_models import EssenceMeta, PaletteProfile


def test_new_essence_round_trips_through_json():
    meta = EssenceMeta(
        id="abc123",
        name="Test",
        created_at="2026-01-01T00:00:00+00:00",
        technique="instantstyle-sdxl-controlnet-v1",
        color=(200, 30, 30),
        version=2,
        palette=PaletteProfile(dominant_colors=[], mean_saturation=0.5, mean_luminance=0.5, temperature=0.0, contrast=0.3),
    )
    loaded = EssenceMeta.model_validate(json.loads(meta.model_dump_json()))
    assert loaded == meta


def test_legacy_meta_without_version_or_analysis_fields_loads_cleanly():
    # Exactly what a pre-structured-Essence meta.json looked like.
    legacy = {
        "id": "legacy1",
        "name": "Old One",
        "technique": "instantstyle-sdxl-controlnet-v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "color": [180, 90, 40],
    }
    meta = EssenceMeta.model_validate(legacy)
    assert meta.version == 1
    assert meta.palette is None
    assert meta.texture is None
    assert meta.stroke is None
    assert meta.style_statistics is None


def test_list_essences_loads_a_legacy_essence_without_crashing(monkeypatch, tmp_path):
    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    one = essences_dir / "legacy1"
    one.mkdir()
    (one / "meta.json").write_text(json.dumps({
        "id": "legacy1",
        "name": "Old One",
        "technique": "instantstyle-sdxl-controlnet-v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "color": [180, 90, 40],
    }))
    # No thumbnail.png — list_essences must also tolerate that (matches
    # existing behavior, unrelated to this schema change).

    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)

    result = essence_store.list_essences()
    assert len(result) == 1
    assert result[0]["id"] == "legacy1"
    assert result[0]["version"] == 1
    assert result[0]["analysis"]["palette"] is None
    assert result[0]["analysis"]["stroke"] is None
    assert result[0]["thumbnail"] is None


def test_list_essences_loads_a_new_style_essence(monkeypatch, tmp_path):
    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    one = essences_dir / "new1"
    one.mkdir()
    meta = EssenceMeta(
        id="new1",
        name="New One",
        created_at="2026-01-01T00:00:00+00:00",
        technique="instantstyle-sdxl-controlnet-v1",
        color=(10, 20, 30),
        version=2,
        palette=PaletteProfile(dominant_colors=[], mean_saturation=0.4, mean_luminance=0.6, temperature=-0.2, contrast=0.5),
    )
    (one / "meta.json").write_text(meta.model_dump_json())

    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)

    result = essence_store.list_essences()
    assert len(result) == 1
    assert result[0]["version"] == 2
    assert result[0]["analysis"]["palette"]["mean_saturation"] == pytest.approx(0.4)


def test_stroke_field_round_trips_through_disk(monkeypatch, tmp_path):
    # _run_analyzers persists the raw stroke field (stroke_field.npz)
    # alongside the four scalar profiles; load_stroke_field is generation.py's
    # own read path for it at apply time (see stroke_texture.py). Exercises
    # that write/read pair directly, without the SDXL pipeline -- neither
    # function touches it.
    from PIL import Image, ImageDraw

    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    out_dir = essences_dir / "abc123"
    out_dir.mkdir()

    # A strongly horizontal-line reference so stroke analysis has a clear,
    # non-degenerate field to persist.
    src = Image.new("L", (200, 200), 255)
    d = ImageDraw.Draw(src)
    for y in range(10, 200, 14):
        d.line([(0, y), (200, y)], fill=0, width=3)
    src = src.convert("RGB")

    analysis = essence_store._run_analyzers(src, out_dir)
    assert analysis["stroke"] is not None
    assert analysis["stroke"].field_path == "stroke_field.npz"
    assert (out_dir / "stroke_field.npz").exists()

    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)
    loaded = essence_store.load_stroke_field("abc123")
    assert loaded is not None
    theta, coherence = loaded
    assert theta.shape == coherence.shape
    assert theta.shape[0] > 1  # a real grid, not a scalar/degenerate shape
    # A strongly horizontal reference should have produced a strongly
    # directional profile -- confirms the persisted grid is the real
    # analysis output, not e.g. an empty/zeroed placeholder.
    assert analysis["stroke"].directionality > 0.7


def test_load_stroke_field_returns_none_when_absent(monkeypatch, tmp_path):
    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    (essences_dir / "no-stroke").mkdir()
    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)
    assert essence_store.load_stroke_field("no-stroke") is None


def test_load_texture_source_returns_none_when_absent(monkeypatch, tmp_path):
    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    (essences_dir / "no-texture").mkdir()
    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)
    assert essence_store.load_texture_source("no-texture") is None


def test_load_texture_source_round_trips_a_saved_swatch(monkeypatch, tmp_path):
    from PIL import Image

    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    out_dir = essences_dir / "abc123"
    out_dir.mkdir()
    Image.new("RGB", (40, 30), (10, 20, 30)).save(out_dir / "texture_source.png")

    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)
    loaded = essence_store.load_texture_source("abc123")
    assert loaded is not None
    assert loaded.size == (40, 30)
    assert loaded.mode == "RGB"


def test_meta_response_reports_has_texture_source(monkeypatch, tmp_path):
    # extract_essence needs the SDXL pipeline (not exercised here); this
    # checks the has_texture_source flag _meta_response derives -- same
    # file-existence pattern as media.py's has_gif -- directly.
    essences_dir = tmp_path / "essences"
    essences_dir.mkdir()
    with_texture = essences_dir / "with-texture"
    with_texture.mkdir()
    (with_texture / "texture_source.png").write_bytes(b"not a real png, existence is all that matters here")
    without_texture = essences_dir / "without-texture"
    without_texture.mkdir()

    meta_kwargs = dict(
        name="Test",
        created_at="2026-01-01T00:00:00+00:00",
        technique="instantstyle-sdxl-controlnet-v1",
        color=(10, 20, 30),
        version=2,
    )
    monkeypatch.setattr(paths, "essences_dir", lambda: essences_dir)

    with_meta = EssenceMeta(id="with-texture", **meta_kwargs)
    (with_texture / "meta.json").write_text(with_meta.model_dump_json())
    without_meta = EssenceMeta(id="without-texture", **meta_kwargs)
    (without_texture / "meta.json").write_text(without_meta.model_dump_json())

    result = {e["id"]: e for e in essence_store.list_essences()}
    assert result["with-texture"]["has_texture_source"] is True
    assert result["without-texture"]["has_texture_source"] is False
