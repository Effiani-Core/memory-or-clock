"""Optional local-only frozen LLM. Only the public observation vector is encoded."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .components import Component, REGISTRY, register_component

DEFAULT_MODEL = "/home/researcher/lab-state/models/Qwen2.5-1.5B-Instruct-989aa798"


def local_identity(directory):
    path = Path(directory).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Frozen model path must be a local directory")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Frozen model requires a local manifest.json with revision and file hashes")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not all(isinstance(manifest.get(key), str) and manifest[key].strip()
               for key in ("model_id", "revision")):
        raise ValueError("Frozen model manifest must declare model_id and revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen model manifest requires file identities")
    # Cover every ordinary model file, including tokenizer data, generation
    # configuration and shard indexes. Download caches are not model inputs.
    present = {file.relative_to(path).as_posix() for file in path.rglob("*")
               if file.is_file() and file.name != "manifest.json"
               and not any(part.startswith(".") for part in file.relative_to(path).parts)}
    if not any(name.endswith(".safetensors") for name in present):
        raise ValueError("Local frozen model has no safetensors weights")
    if not any(name in present for name in ("tokenizer.json", "vocab.json", "vocab.txt")) \
            and not any(name.endswith((".model", ".tiktoken")) for name in present):
        raise ValueError("Local frozen model has no tokenizer vocabulary/data")
    required = {"config.json", "tokenizer_config.json"} | present
    if not required <= set(files):
        raise ValueError("Frozen model manifest does not cover its loading configuration/tokenizer/weights")
    for name, expected in files.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Manifest file escapes local model directory")
        if not isinstance(expected, dict) or type(expected.get("bytes")) is not int or expected["bytes"] < 0:
            raise ValueError(f"Invalid frozen model file size identity: {name}")
        expected_hash = expected.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
                char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError(f"Invalid frozen model file checksum identity: {name}")
        try:
            file = (path / name).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"Frozen model file is missing: {name}") from exc
        if not file.is_relative_to(path) or not file.is_file():
            raise ValueError("Manifest file escapes local model directory")
        if file.stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen model file size changed: {name}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Frozen model file checksum changed: {name}")
    return {"path": str(path), "model_id": manifest["model_id"],
            "revision": manifest["revision"], "verified_files": len(files),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


class FrozenLLM(Component):
    """Public numerical observations -> textual tokens -> mean last-layer representation.

    No external memory, retrieval, hidden environment state, or parameter updates.
    Frozen pretrained weights are referenced, not duplicated, in checkpoints.
    """
    def __init__(self, input_dim, model_dir=DEFAULT_MODEL, output_dim=None, max_tokens=512,
                 revision=None, device="cuda", dtype="bfloat16"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Configured frozen LLM requires CUDA; no silent CPU fallback")
        self.identity = local_identity(model_dir)
        if revision is not None and revision != self.identity["revision"]:
            raise ValueError("Frozen model revision does not match requested version")
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        width = config.hidden_size
        if output_dim is not None and output_dim != width:
            raise ValueError(f"LLM hidden width is {width}; declare an explicit MLP adapter")
        super().__init__(input_dim, width)
        self.max_tokens = int(max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if str(device).startswith("cuda"):
            free_bytes, _ = torch.cuda.mem_get_info()
            weight_bytes = sum(path.stat().st_size for path in Path(model_dir).glob("*.safetensors"))
            if weight_bytes <= 0:
                raise ValueError("Local frozen model has no safetensors weights")
            if free_bytes < weight_bytes + 1024**3:
                raise RuntimeError("Insufficient free VRAM for frozen weights plus 1 GiB runtime reserve")
        self._llm = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), device_map={"": device}, low_cpu_mem_usage=True)
        self._llm.requires_grad_(False)
        self._llm.eval()

    def train(self, mode=True):
        super().train(mode)
        self._llm.eval()
        return self

    def _inputs(self, x, action_request=False):
        texts = []
        for row in x.detach().cpu().tolist():
            text = "Visible egocentric observation, tile/color values normalized by the adapter: "
            text += ",".join(f"{value:.4f}" for value in row)
            if action_request:
                text += "\nActions: 0 forward, 1 turn right, 2 turn left, 3 pick up, 4 put down, 5 toggle. Choose one action. Reply with its number only."
            messages = [{"role": "user", "content": text}]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        if inputs.input_ids.shape[1] > self.max_tokens:
            raise ValueError("Frozen encoder context limit exceeded; observations were not truncated")
        return inputs.to(next(self._llm.parameters()).device)

    def forward(self, x, state):
        self.validate(x, state)
        inputs = self._inputs(x)
        with torch.no_grad():
            hidden = self._llm.model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            features = (hidden * mask).sum(1) / mask.sum(1)
        return features.float().to(x.device), {}, {"frozen": True, "encoded_tokens": int(inputs.attention_mask.sum())}

    def action_logits(self, x):
        inputs = self._inputs(x, action_request=True)
        indices = []
        for action in range(6):
            tokens = self.tokenizer.encode(str(action), add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError("Direct action baseline requires single-token action labels")
            indices.append(tokens[0])
        with torch.no_grad():
            logits = self._llm(**inputs).logits
            last = inputs.attention_mask.sum(1) - 1
            logits = logits[torch.arange(x.shape[0], device=logits.device), last][:, indices]
        return logits.float().to(x.device)

    def external_checkpoint_reference(self):
        return {"prefixes": ["_llm."], "identity": self.identity}

    def contract(self):
        result = super().contract()
        result.update(checkpoint={"version": 1, "strict_shapes": True,
                                  "external_reference": self.identity},
                      instrumentation=["frozen", "encoded_tokens"],
                      learning_operations=[])
        return result


class FrozenLLMPolicy(torch.nn.Module):
    checkpoint_version = 1
    def __init__(self, config, input_dim, action_count):
        super().__init__()
        if action_count != 6:
            raise ValueError("Direct LLM baseline currently supports XLand's six action labels")
        self.config, self.input_dim, self.action_count = dict(config), input_dim, action_count
        self.encoder = FrozenLLM(input_dim, **config.get("params", {}))
        self.growth_events = []
    def initial_state(self, batch_size, device):
        return {}
    def forward(self, obs, state=None):
        if state not in ({}, None):
            raise ValueError("Frozen direct baseline has no recurrent state")
        return {"logits": self.encoder.action_logits(obs), "value": torch.zeros(obs.shape[0], device=obs.device),
                "state": {}, "signals": {"frozen_llm": {"frozen": True}}}
    def parameter_counts(self):
        return {"trainable": 0, "frozen": sum(p.numel() for p in self.parameters()),
                "total": sum(p.numel() for p in self.parameters())}
    def contract(self):
        return {"config": self.config, "components": {"encoder": self.encoder.contract()},
                "parameters": self.parameter_counts(), "input_dim": self.input_dim,
                "action_count": self.action_count, "checkpoint_version": 1}


def register_frozen():
    if "frozen_llm" not in REGISTRY:
        register_component("frozen_llm", FrozenLLM)
