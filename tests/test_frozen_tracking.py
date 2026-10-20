"""Tiny fixtures: no actual LLM load, downloads, GPU or existing MLflow writes."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest


@pytest.fixture
def lab_tmp():
    parent = Path(__file__).resolve().parents[1] / "runs" / "test-fixtures"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frozen-tracking-", dir=parent) as directory:
        yield Path(directory)


def digest(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def model_dir(lab_tmp):
    path = lab_tmp / "fake-model"
    path.mkdir()
    contents = {
        "config.json": b'{"hidden_size":4}',
        "tokenizer_config.json": b'{"model_max_length":16}',
        "tokenizer.json": b'{"fake_tokenizer":true}',
        "vocab.json": b'{"word":0}',
        "merges.txt": b"#fake merges\n",
        "generation_config.json": b'{"max_length":16}',
        "model.safetensors": b"FAKE WEIGHTS: identity-only fixture, never loaded",
    }
    for name, payload in contents.items():
        (path / name).write_bytes(payload)
    manifest = {"model_id": "fixture/tiny", "revision": "fixture-revision-1",
                "files": {name: digest(payload) for name, payload in contents.items()}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def edit_manifest(path, update):
    file = path / "manifest.json"
    data = json.loads(file.read_text())
    update(data)
    file.write_text(json.dumps(data))


def test_local_identity_uses_all_hashes_and_revision(model_dir):
    from ucl.frozen import local_identity
    identity = local_identity(model_dir)
    assert identity["model_id"] == "fixture/tiny"
    assert identity["revision"] == "fixture-revision-1"
    assert identity["verified_files"] == 7
    assert identity["manifest_sha256"] == hashlib.sha256((model_dir / "manifest.json").read_bytes()).hexdigest()
    assert identity == local_identity(model_dir)


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "tokenizer.json", "generation_config.json"])
def test_same_length_changes_fail_checksum(model_dir, name):
    from ucl.frozen import local_identity
    file = model_dir / name
    original = file.read_bytes()
    file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert file.stat().st_size == len(original)
    with pytest.raises(ValueError, match="checksum changed"):
        local_identity(model_dir)


@pytest.mark.parametrize("name", ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "generation_config.json", "model.safetensors"])
def test_manifest_must_cover_loading_files(model_dir, name):
    from ucl.frozen import local_identity
    edit_manifest(model_dir, lambda manifest: manifest["files"].pop(name))
    with pytest.raises(ValueError, match="does not cover"):
        local_identity(model_dir)


def test_manifest_covers_shard_index_and_new_tokenizer_assets(model_dir):
    from ucl.frozen import local_identity
    (model_dir / "model.safetensors.index.json").write_text('{}')
    with pytest.raises(ValueError, match="does not cover"):
        local_identity(model_dir)


@pytest.mark.parametrize("field,value", [("revision", ""), ("revision", "  "), ("model_id", None)])
def test_manifest_requires_meaningful_version(model_dir, field, value):
    from ucl.frozen import local_identity
    edit_manifest(model_dir, lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError, match="model_id and revision"):
        local_identity(model_dir)


@pytest.mark.parametrize("mode", ["parent", "absolute", "symlink"])
def test_manifest_cannot_escape_directory(model_dir, lab_tmp, mode):
    from ucl.frozen import local_identity
    outside = lab_tmp / "outside.bin"
    outside.write_bytes(b"outside fixture")
    if mode == "parent":
        name = "../outside.bin"
    elif mode == "absolute":
        name = str(outside.resolve())
    else:
        name = "linked.bin"
        (model_dir / name).symlink_to(outside)
    edit_manifest(model_dir, lambda manifest: manifest["files"].update({name: digest(outside.read_bytes())}))
    with pytest.raises(ValueError, match="escapes"):
        local_identity(model_dir)


def test_missing_weights_cannot_pass_identity(model_dir):
    from ucl.frozen import local_identity
    (model_dir / "model.safetensors").unlink()
    edit_manifest(model_dir, lambda manifest: manifest["files"].pop("model.safetensors"))
    with pytest.raises(ValueError, match="no safetensors"):
        local_identity(model_dir)


def test_requested_revision_rejected_before_model_load(model_dir, monkeypatch):
    from ucl.frozen import FrozenLLM
    class NeverLoad:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            pytest.fail("No transformer loader should run for a revision mismatch")
    stub = SimpleNamespace(AutoConfig=NeverLoad, AutoModelForCausalLM=NeverLoad, AutoTokenizer=NeverLoad)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    with pytest.raises(ValueError, match="revision does not match"):
        FrozenLLM(50, model_dir=str(model_dir), revision="wrong-revision", device="cpu")


def test_namespace_rejected_before_network_or_client(lab_tmp, monkeypatch):
    import ucl.tracking as tracking
    def prohibited(*args, **kwargs):
        pytest.fail("Invalid namespace must fail before touching a server/store")
    monkeypatch.setattr(tracking, "urlopen", prohibited)
    monkeypatch.setattr(tracking, "MlflowClient", prohibited)
    with pytest.raises(ValueError, match="own MLflow experiment namespace"):
        tracking.Tracker(lab_tmp, {"experiment": "Existing Lab"}, "forbidden")
    assert not (lab_tmp / "runs").exists()


class Healthy:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return False


class FakeClient:
    def __init__(self, tracking_uri):
        self.uri = tracking_uri
        self.calls = []
    def get_experiment_by_name(self, name):
        self.calls.append(("get", name))
        return None
    def create_experiment(self, name, artifact_location):
        self.calls.append(("experiment", name, artifact_location))
        return "42"
    def create_run(self, experiment_id, tags):
        self.calls.append(("run", experiment_id, tags))
        return SimpleNamespace(info=SimpleNamespace(run_id="run42"))
    def log_artifacts(self, run_id, directory):
        raise OSError("fixture artifact upload failure")
    def set_terminated(self, run_id, status):
        self.calls.append(("terminated", run_id, status))


def test_healthy_server_uri_preserved_and_new_namespace_only(lab_tmp, monkeypatch):
    import mlflow
    import ucl.tracking as tracking
    original = mlflow.get_tracking_uri()
    monkeypatch.setattr(tracking, "urlopen", lambda *args, **kwargs: Healthy())
    monkeypatch.setattr(tracking, "MlflowClient", FakeClient)
    tracker = tracking.Tracker(lab_tmp, {"uri": "http://127.0.0.1:5000"}, "small-fixture")
    assert not tracker.fallback and tracker.client.uri == "http://127.0.0.1:5000"
    assert mlflow.get_tracking_uri() == original
    assert tracker.client.calls[0] == ("get", "Universal Cognitive Lab")
    assert tracker.client.calls[1][1] == "Universal Cognitive Lab"
    assert tracker.client.calls[1][2] == (lab_tmp / "runs/mlartifacts").as_uri()
    assert tracker.reference()["run_id"] == "run42"
    with pytest.raises(OSError, match="artifact upload"):
        tracker.finish(lab_tmp)
    assert tracker.client.calls[-1] == ("terminated", "run42", "FAILED")


def test_unavailable_server_uses_real_isolated_store_without_global_redirect(lab_tmp, monkeypatch):
    import mlflow
    import ucl.tracking as tracking
    original = mlflow.get_tracking_uri()
    def unavailable(*args, **kwargs): raise URLError("fixture offline; no network requested")
    monkeypatch.setattr(tracking, "urlopen", unavailable)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracker = tracking.Tracker(lab_tmp, {"uri": "http://127.0.0.1:5000"}, "offline-fixture")
    assert tracker.fallback
    assert tracker.uri == (lab_tmp / "runs/mlruns-isolated").as_uri()
    assert mlflow.get_tracking_uri() == original
    tracker.params({"seed": 7, "architecture": {"type": "fixture"}})
    tracker.metrics({"reward": 0.5, "nan": float("nan"), "inf": float("inf"), "boolean": True}, step=2)
    artifact = lab_tmp / "artifacts"
    artifact.mkdir()
    (artifact / "evidence.json").write_text('{"fixture": true}')
    tracker.finish(artifact)
    run = tracker.client.get_run(tracker.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "7"
    assert run.data.metrics == {"reward": 0.5}
    history = tracker.client.get_metric_history(tracker.run_id, "reward")
    assert history[0].step == 2
    assert tracker.client.get_experiment(tracker.experiment_id).name == "Universal Cognitive Lab"
    assert [item.path for item in tracker.client.list_artifacts(tracker.run_id)] == ["evidence.json"]
    assert tracker.reference()["url"] is None
