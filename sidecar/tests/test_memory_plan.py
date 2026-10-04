"""Restyle-performance contract: memory plan, progress tracker, cached prompt
embeds, /apply/progress. No models, no GPU — stubs only."""
import generation
import pipeline_manager
from pipeline_manager import RESIDENT_MIN_FREE_BYTES, choose_memory_plan

GB = 1024**3


def test_plan_cpu_when_no_cuda():
    assert choose_memory_plan(0, 0, "cpu") == "cpu"
    assert choose_memory_plan(100 * GB, 100 * GB, "cpu") == "cpu"


def test_plan_resident_at_and_above_threshold():
    assert choose_memory_plan(RESIDENT_MIN_FREE_BYTES, 12 * GB, "cuda") == "resident"
    assert choose_memory_plan(20 * GB, 24 * GB, "cuda") == "resident"


def test_plan_offload_below_threshold():
    assert choose_memory_plan(RESIDENT_MIN_FREE_BYTES - 1, 12 * GB, "cuda") == "offload"
    assert choose_memory_plan(3 * GB, 8 * GB, "cuda") == "offload"


def test_plan_offload_when_total_too_small_even_if_free_claims_enough():
    assert choose_memory_plan(RESIDENT_MIN_FREE_BYTES, 8 * GB, "cuda") == "offload"


def test_tracker_idle_snapshot():
    assert generation.ProgressTracker().snapshot() == {"active": False}


def test_tracker_rolling_median_and_slow_flag():
    t = generation.ProgressTracker()
    t.begin(pass_count=2)
    t.start_pass(now=0.0)
    # durations 1, 1, 20 -> median 1 (outlier ignored); window keeps last 3
    t.on_step(1, 10, now=1.0)
    t.on_step(2, 10, now=2.0)
    t.on_step(3, 10, now=22.0)
    snap = t.snapshot()
    assert snap["active"] and snap["pass_index"] == 1 and snap["pass_count"] == 2
    assert snap["step"] == 3 and snap["total_steps"] == 10
    assert snap["sec_per_step"] == 1.0 and snap["slow"] is False
    # next two slow steps push the median over the threshold
    t.on_step(4, 10, now=32.0)
    t.on_step(5, 10, now=42.0)  # last 3 = 20, 10, 10 -> median 10
    snap = t.snapshot()
    assert snap["sec_per_step"] == 10.0 and snap["slow"] is True


def test_tracker_slow_threshold_is_strict():
    t = generation.ProgressTracker()
    t.begin()
    t.start_pass(now=0.0)
    t.on_step(1, 5, now=generation.SLOW_STEP_S)
    assert t.snapshot()["slow"] is False


def test_tracker_reset_and_finish():
    t = generation.ProgressTracker()
    t.begin(pass_count=2)
    t.start_pass(now=0.0)
    t.on_step(1, 10, now=9.0)
    assert t.snapshot()["slow"] is True
    t.begin(pass_count=1)  # new apply resets everything
    snap = t.snapshot()
    assert snap["active"] and snap["step"] == 0 and snap["sec_per_step"] is None and snap["slow"] is False
    t.start_pass(now=0.0)
    t.start_pass(now=1.0)
    assert t.snapshot()["pass_index"] == 2
    t.finish()
    assert t.snapshot() == {"active": False}


def test_run_generation_uses_cached_embeds(monkeypatch):
    cached = {
        "prompt_embeds": "pe",
        "negative_prompt_embeds": "npe",
        "pooled_prompt_embeds": "ppe",
        "negative_pooled_prompt_embeds": "nppe",
    }
    monkeypatch.setattr(pipeline_manager, "_prompt_embeds", cached)
    seen = {}

    class Result:
        images = ["img"]

    class StubPipe:
        num_timesteps = 4

        def __call__(self, **kwargs):
            seen.update(kwargs)
            cb = kwargs["callback_on_step_end"]
            for i in range(4):
                cb(self, i, None, {})
            return Result()

    generation.progress.begin()
    out = generation._run_generation(StubPipe(), ["emb"], "working", 0.8, 0.85, 30)
    assert out == "img"
    for k, v in cached.items():
        assert seen[k] == v
    assert "prompt" not in seen and "negative_prompt" not in seen
    assert seen["ip_adapter_image_embeds"] == ["emb"]
    snap = generation.progress.snapshot()
    assert snap["step"] == 4 and snap["total_steps"] == 4
    generation.progress.finish()


def test_run_generation_falls_back_to_strings_without_cache(monkeypatch):
    monkeypatch.setattr(pipeline_manager, "_prompt_embeds", None)
    seen = {}

    class Result:
        images = ["img"]

    class StubPipe:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return Result()

    generation._run_generation(StubPipe(), [], "w", 0.8, 0.85, 30)
    assert seen["prompt"] == "" and seen["negative_prompt"] == generation.NEGATIVE_PROMPT
    generation.progress.finish()


def test_apply_progress_endpoint_idle():
    import app

    generation.progress.finish()
    assert app.apply_progress_endpoint() == {"active": False}


def test_apply_request_blend_default_is_none():
    import app

    assert app.ApplyRequest(essence_id="x", image_path="y").blend_mode == "none"


def test_health_model_status_exposes_memory_plan():
    assert "memory_plan" in pipeline_manager.status()
