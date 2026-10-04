import threading
import time

import model_downloads as md


def _reset(monkeypatch, items):
    monkeypatch.setattr(md, "_items", items)
    monkeypatch.setattr(md, "_state", "idle")
    monkeypatch.setattr(md, "_error", None)
    monkeypatch.setattr(md, "_thread", None)
    monkeypatch.setattr(md, "_pause_requested", False)
    md._samples.clear()
    monkeypatch.setattr(md, "_cached", lambda item: False)
    monkeypatch.setattr(md, "_write_manifest", lambda item: None)
    monkeypatch.setattr(md, "_measure", lambda item: (setattr(item, "total", 100), setattr(item, "files", ["f"])))


def _wait(pred, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_patterns_skip_large_unneeded_files():
    sdxl = next(i for i in md.build_items(True) if i.id == "sdxl")
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in sdxl.allow
    assert not any("sd_xl_base" in p or "onnx" in p for p in sdxl.allow)
    cpu = next(i for i in md.build_items(False) if i.id == "sdxl")
    assert "unet/diffusion_pytorch_model.safetensors" in cpu.allow


def test_runs_all_items_to_ready(monkeypatch):
    _reset(monkeypatch, [md.Item("a", "A", "r/a", []), md.Item("b", "B", "r/b", [])])
    monkeypatch.setattr(md, "_download_item", lambda item: setattr(item, "done_bytes", 100))
    md.start()
    assert _wait(md.is_ready)
    st = md.status()
    assert st["downloaded_bytes"] == st["total_bytes"] == 200
    assert all(i["state"] == "done" for i in st["items"])


def test_cached_items_count_as_done(monkeypatch):
    _reset(monkeypatch, [md.Item("a", "A", "r/a", [])])
    monkeypatch.setattr(md, "_cached", lambda item: True)

    def boom(item):
        raise AssertionError("should not download")

    monkeypatch.setattr(md, "_download_item", boom)
    md.start()
    assert _wait(md.is_ready)


def test_pause_then_resume(monkeypatch):
    _reset(monkeypatch, [md.Item("a", "A", "r/a", [])])
    gate = threading.Event()
    attempts = []

    def fake(item):
        attempts.append(1)
        if len(attempts) == 1:
            gate.wait(2)
            if md._pause_requested:
                raise md._Paused()
        item.done_bytes = 100

    monkeypatch.setattr(md, "_download_item", fake)
    md.start()
    md.pause()
    gate.set()
    assert _wait(lambda: md.status()["state"] == "paused")
    md.resume()
    assert _wait(md.is_ready)
    assert len(attempts) == 2


def test_error_then_retry(monkeypatch):
    _reset(monkeypatch, [md.Item("a", "A", "r/a", [])])
    calls = []

    def fake(item):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("network down")

    monkeypatch.setattr(md, "_download_item", fake)
    md.start()
    assert _wait(lambda: md.status()["state"] == "error")
    assert "network down" in md.status()["error"]
    md.resume()
    assert _wait(md.is_ready)


def _disk_setup(monkeypatch, tmp_path, free, present=0, total=1000):
    _reset(monkeypatch, [md.Item("a", "A", "r/a", [])])
    monkeypatch.setattr(md, "_measure", lambda item: (setattr(item, "total", total), setattr(item, "files", ["f"])))
    monkeypatch.setattr(md.paths, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(md, "DISK_SAFETY_MARGIN", 500)
    monkeypatch.setattr(md, "_present_bytes", lambda item: present)
    monkeypatch.setattr(md, "_disk", None)
    monkeypatch.setattr(md.shutil, "disk_usage", lambda p: type("U", (), {"free": free})())
    monkeypatch.setattr(md, "_download_item", lambda item: setattr(item, "done_bytes", total))


def test_refuses_when_disk_short(monkeypatch, tmp_path):
    _disk_setup(monkeypatch, tmp_path, free=1499)
    md.start()
    assert _wait(lambda: md.status()["state"] == "insufficient_disk")
    st = md.status()
    assert st["required_bytes"] == 1500 and st["free_bytes"] == 1499
    assert st["path"] == str(tmp_path)
    assert all(i["state"] == "pending" for i in st["items"])


def test_allows_when_enough_disk(monkeypatch, tmp_path):
    _disk_setup(monkeypatch, tmp_path, free=1500)
    md.start()
    assert _wait(md.is_ready)
    assert md.status()["required_bytes"] is None


def test_present_bytes_reduce_requirement(monkeypatch, tmp_path):
    _disk_setup(monkeypatch, tmp_path, free=900, present=600)
    md.start()  # needs 400 + 500 margin = 900
    assert _wait(md.is_ready)


def test_resume_rechecks_disk(monkeypatch, tmp_path):
    _disk_setup(monkeypatch, tmp_path, free=100)
    md.start()
    assert _wait(lambda: md.status()["state"] == "insufficient_disk")
    monkeypatch.setattr(md.shutil, "disk_usage", lambda p: type("U", (), {"free": 10_000})())
    md.resume()
    assert _wait(md.is_ready)
