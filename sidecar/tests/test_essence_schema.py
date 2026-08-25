"""Essence schema round-trip and backward compatibility (spec Phase 20).
Exercises essence_models.EssenceMeta directly, plus essence.list_essences()
against a temp essences_dir — list_essences() doesn't touch the SDXL
pipeline (only extract_essence does), so this needs no GPU.
"""
import json

import pytest

import essence
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

    result = essence.list_essences()
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

    result = essence.list_essences()
    assert len(result) == 1
    assert result[0]["version"] == 2
    assert result[0]["analysis"]["palette"]["mean_saturation"] == pytest.approx(0.4)
