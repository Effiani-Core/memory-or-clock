"""One local experiment runner, independent of the selected neural architecture."""
from __future__ import annotations
from . import runtime  # noqa: F401 -- CPU JAX/offline defaults precede scientific imports.
import copy
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from .checkpoints import build_policy, load_checkpoint, restore_rng, save_checkpoint, validate_selected_architecture
from .envs import make_env
from .evaluation import evaluate
from .growth import GrowthPolicy, make_growth_policy
from .learning import A2CTrainer, parameter_digest, seed_everything
from .resources import ResourceGuard
from .tasks import load_tasks
from .tracking import Tracker

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def snapshot_sources(output):
    identities = {}
    files = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("configs").glob("*.json"),
             ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for file in files:
        relative = file.relative_to(ROOT)
        data = file.read_bytes()
        identities[str(relative)] = hashlib.sha256(data).hexdigest()
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"files": identities, "identity": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()}


def new_output(label):
    return ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + label)


def _evaluation_summary(result):
    return {key:value for key,value in result.items() if key != "records"}


def run_experiment(config, output=None, *, mode="train", checkpoint=None, task_id=None, curriculum=None):
    config = copy.deepcopy(config)
    if mode not in ("train", "evaluate"):
        raise ValueError("Unknown experiment mode")
    for module in config.get("plugins", []):
        if not isinstance(module, str):
            raise ValueError("Plugins must be explicit importable local module names")
        importlib.import_module(module)
    output = Path(output or new_output(config["architecture_id"])).resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Experiment output must remain inside this new laboratory's runs/")
    output.mkdir(parents=True, exist_ok=False)
    lock = (ROOT / "runs/.execution.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another laboratory execution is active; models run sequentially") from exc
    tracker = model = trainer = None
    started = time.perf_counter()
    summary = {"status": "RUNNING", "architecture_id": config["architecture_id"],
               "config": config, "checkpoint": str(checkpoint) if checkpoint else None,
               "mode": mode, "claim": config.get("claim"), "output": str(output)}
    guard = ResourceGuard(config.get("resources"), output)
    try:
        guard.check()
        tasks = load_tasks(ROOT / config.get("task_manifest", "configs/tasks.json"))
        task_id = task_id or config["task_id"]
        if task_id not in tasks["tasks"]:
            raise ValueError(f"Unknown task {task_id}")
        task = tasks["tasks"][task_id]
        if mode == "train" and task["split"] != "development":
            raise ValueError("Training is restricted to development tasks")
        if config.get("availability") == "unavailable":
            raise RuntimeError(config["unavailable_reason"])
        seed_everything(int(config["seed"]))
        write_json(output / "config.json", config)
        write_json(output / "tasks_manifest.json", tasks)
        sources = snapshot_sources(output)
        write_json(output / "source_identity.json", sources)
        versions = {name:importlib.metadata.version(name) for name in ["torch","numpy","xminigrid","jax","jaxlib","flax","mlflow-skinny","transformers","streamlit"]}
        write_json(output / "versions.json", versions)
        env = make_env(task)
        if checkpoint:
            model, optimizer, payload = load_checkpoint(checkpoint, config.get("device", "cpu"))
            validate_selected_architecture(config["architecture"], model.config)
            if model.input_dim != env.observation_dim or model.action_count != env.action_count:
                raise ValueError("Checkpoint/environment input or action dimensions differ")
            if mode == "train":
                if payload["trainer"] is None:
                    raise ValueError("This checkpoint contains no training state for resume")
                if payload["metadata"].get("task_id") != task_id:
                    raise ValueError("Resume must restore the same current task; use a curriculum to transfer")
                trainer = A2CTrainer(model, env, config["seed"], payload["trainer"]["config"], optimizer, seed_pool=task["seeds"])
                trainer.restore_state(payload["trainer"])
                restore_rng(payload)
        elif config["architecture"]["kind"] != "random":
            model = build_policy(config["architecture"], env.observation_dim, env.action_count, config.get("device", "cpu"))
        if model is None:
            counts = {"trainable":0,"frozen":0,"total":0}
        else:
            counts = model.parameter_counts()
            write_json(output / "component_contracts.json", model.contract())
        summary["initial_parameters"] = counts
        summary["source_identity"] = sources["identity"]
        summary["initial_trainable_sha256"] = parameter_digest(model, frozen=False) if model else None
        frozen_before = parameter_digest(model, frozen=True) if model else None
        summary["initial_frozen_sha256"] = frozen_before
        tracker = Tracker(ROOT, config.get("tracking", {}), config["architecture_id"] + " - " + mode)
        summary["mlflow"] = tracker.reference()
        tracker.params({"architecture_id":config["architecture_id"],"environment_id":task["env_id"],
                        "task_family":task["family"],"seed":config["seed"],
                        "algorithm":"a2c" if mode=="train" and counts["trainable"] else "evaluation_only",
                        "architecture":config["architecture"],"source_identity":sources["identity"],
                        "trainable_parameters":counts["trainable"],"frozen_parameters":counts["frozen"],
                        "model_size_bytes":sum(p.numel()*p.element_size() for p in model.parameters()) if model else 0,
                        "evaluation_task":config.get("evaluation_task"),"training_exposure_requested":config.get("train_steps",0)})
        write_json(output / "progress.json", summary)
        algorithm = {k:v for k,v in config.get("algorithm",{}).items() if k != "name"}
        if config.get("algorithm",{}).get("name","a2c") != "a2c":
            raise ValueError("Only implemented algorithm a2c is available")
        growth = GrowthPolicy.from_state_dict(payload["growth_policy"]) if checkpoint and mode=="train" and payload.get("growth_policy") else make_growth_policy(config.get("growth"))
        should_train = mode == "train" and counts["trainable"] > 0
        if should_train and trainer is None:
            trainer = A2CTrainer(model, env, task["seeds"][0] + int(config["seed"]), algorithm, seed_pool=task["seeds"])
        training_start_steps = trainer.total_steps if trainer else 0
        if model:
            save_checkpoint(output / "checkpoints/initial.pt", model, trainer, metadata={"task_id":task_id}, growth=growth)
        evaluations, growth_events, stage_results = [], [], []
        if curriculum:
            if curriculum not in tasks["curricula"]:
                raise ValueError(f"Unknown curriculum {curriculum}")
            stages = copy.deepcopy(tasks["curricula"][curriculum]["stages"])
        else:
            stages = copy.deepcopy(config.get("stages", [{"task_id":task_id,"train_steps":config.get("train_steps",0),"purpose":"train"}]))
        if mode == "evaluate":
            stages = [{**stage,"train_steps":0} for stage in stages]
        best_score = float("-inf")
        initial_validation = tasks["tasks"][config.get("evaluation_task",task_id)]
        eval_seeds = config.get("evaluation_seeds", initial_validation["evaluation_seeds"][:2])
        if mode == "train":
            if initial_validation["split"] != "validation":
                raise ValueError("Checkpoint selection requires a validation task; heldout data cannot select weights")
            initial = evaluate(model, initial_validation, eval_seeds, architecture_id=config["architecture_id"])
            evaluations.append({"phase":"initial_validation", "train_step":trainer.total_steps if trainer else 0, **_evaluation_summary(initial)})
            best_score = initial["success_rate"]
            if model:
                save_checkpoint(output / "checkpoints/best.pt", model, trainer, metadata={"task_id":task_id,"selection_split":"validation","score":best_score}, growth=growth)
        current_task = task_id
        with (output / "metrics.jsonl").open("x") as metrics:
            for index, stage in enumerate(stages):
                guard.check()
                selected = tasks["tasks"][stage["task_id"]]
                train_steps = int(stage.get("train_steps",0)) if should_train else 0
                if train_steps < 0 or train_steps and selected["split"] != "development":
                    raise ValueError("Training exposure must be nonnegative and only on development tasks")
                if trainer and train_steps > 0 and current_task != stage["task_id"]:
                    trainer.set_task(make_env(selected), selected["seeds"][0] + int(config["seed"]), seed_pool=selected["seeds"])
                    current_task = stage["task_id"]
                stage_start = trainer.total_steps if trainer else 0
                stage_training_seeds = []
                while trainer and trainer.total_steps < stage_start + train_steps:
                    resource = guard.check()
                    batch_steps = min(trainer.config["rollout_steps"], stage_start + train_steps-trainer.total_steps)
                    row = trainer.update(batch_steps)
                    stage_training_seeds.extend(row.get("training_seeds", []))
                    row.update(task_id=current_task, stage=index, resource=resource)
                    if growth.observe(trainer.updates, row["rollout_reward_mean"]):
                        try:
                            trainer.state, event = model.grow(growth.node,growth.add_count,trainer.optimizer,trainer.state)
                            event.update(step=trainer.total_steps,update=trainer.updates,policy=growth.kind,
                                         trigger_score=row["rollout_reward_mean"])
                            growth.record_growth(trainer.updates)
                            growth_events.append(event)
                        except ValueError as exc:
                            growth_events.append({"step":trainer.total_steps,"update":trainer.updates,
                                                  "policy":growth.kind,"rejected":str(exc)})
                    metrics.write(json.dumps(row,allow_nan=False)+"\n");metrics.flush()
                    tracker.metrics(row,trainer.total_steps,"train.")
                    tracker.metrics(resource,trainer.total_steps,"resources.")
                    write_json(output / "progress.json",{**summary,"stage":index,"task_id":current_task,
                               "training_steps":trainer.total_steps,"last_training":row,
                               "growth_events":growth_events})
                seeds = [config["seed"]] if mode == "evaluate" else stage.get("evaluation_seeds",selected["evaluation_seeds"][:2])
                recorded = output / "episodes" / f"stage-{index}" if mode == "evaluate" else None
                result = evaluate(model,selected,seeds,output=recorded,architecture_id=config["architecture_id"])
                stage_result={"stage":index,"purpose":stage.get("purpose","evaluation"),
                              "training_steps_before":stage_start,"training_steps_after":trainer.total_steps if trainer else 0,
                              "training_seed_base":trainer.seed if trainer and train_steps > 0 else None,
                              "training_seeds_by_interaction":stage_training_seeds,
                              "training_seed_pool":list(selected["seeds"]) if train_steps > 0 else [],
                              **_evaluation_summary(result)}
                stage_results.append(stage_result)
                tracker.metrics(result,trainer.total_steps if trainer else index,f"stage_{index}.")
                if model and trainer:
                    validation=evaluate(model,initial_validation,eval_seeds,architecture_id=config["architecture_id"])
                    evaluations.append({"phase":"validation","train_step":trainer.total_steps,**_evaluation_summary(validation)})
                    if validation["success_rate"] > best_score:
                        best_score=validation["success_rate"]
                        save_checkpoint(output/"checkpoints/best.pt",model,trainer,
                                        metadata={"task_id":current_task,"selection_split":"validation","score":best_score},growth=growth)
                    save_checkpoint(output/"checkpoints/last.pt",model,trainer,metadata={"task_id":current_task},growth=growth)
        final_task = tasks["tasks"][task_id if mode=="evaluate" else config.get("heldout_task",config.get("evaluation_task",task_id))]
        final_seeds = [config["seed"]] if mode=="evaluate" else final_task["evaluation_seeds"][:len(eval_seeds)]
        final = result if mode == "evaluate" else evaluate(model,final_task,final_seeds,output=output/"episodes",architecture_id=config["architecture_id"])
        tracker.metrics(final,trainer.total_steps if trainer else 0,"evaluation.")
        if model and not (output/"checkpoints/last.pt").exists():
            save_checkpoint(output/"checkpoints/last.pt",model,trainer,metadata={"task_id":current_task},growth=growth)
        if model and not (output/"checkpoints/best.pt").exists():
            save_checkpoint(output/"checkpoints/best.pt",model,trainer,metadata={"task_id":current_task,"selection_split":"none_no_training"},growth=growth)
        after = parameter_digest(model, frozen=False) if model else None
        frozen_after=parameter_digest(model, frozen=True) if model else None
        if frozen_before != frozen_after:
            raise RuntimeError("Frozen parameter tensors changed")
        changed = after != summary["initial_trainable_sha256"]
        if trainer and trainer.total_steps > training_start_steps and not changed:
            raise RuntimeError("Training did not change trainable parameters")
        retention = {}
        learned = next((s for s in stage_results if s.get("purpose")=="learn_A"),None)
        revisited=next((s for s in stage_results if s.get("purpose")=="retention"),None)
        if learned and revisited:
            before, after_score=learned["success_rate"],revisited["success_rate"]
            retention={"task_id":learned["task_id"],"after_learning":before,"after_other_tasks":after_score,
                       "backward_transfer":after_score-before,"forgetting":max(0.0,before-after_score),
                       "interference":max(0.0,before-after_score),
                       "retention_ratio":after_score/before if before>0 else None,
                       "retention_defined":before>0,
                       "note":"No positive initial skill means retention ratio is undefined; zero forgetting alone is not retention evidence."}
            tracker.metrics(retention,prefix="retention.")
        summary.update(status="FINISHED",training_steps=trainer.total_steps if trainer else 0,
                       new_training_steps=(trainer.total_steps-training_start_steps) if trainer else 0,
                       trainable_parameters_changed=changed,frozen_parameters_unchanged=frozen_before==frozen_after,
                       final_trainable_sha256=after,final_frozen_sha256=frozen_after,
                       final_parameters=model.parameter_counts() if model else counts,
                       growth_events=growth_events,evaluations=evaluations,stages=stage_results,
                       retention=retention,metrics=_evaluation_summary(final),
                       checkpoint=str(output/"checkpoints/last.pt") if model else None,
                       elapsed_seconds=time.perf_counter()-started)
        write_json(output/"resources.json",guard.samples)
        write_json(output/"summary.json",summary)
        write_json(output/"progress.json",summary)
        tracker.metrics({"training_steps":summary["training_steps"],"elapsed_seconds":summary["elapsed_seconds"],
                         "expansion_events":sum("rejected" not in e for e in growth_events),
                         "final_trainable_parameters":summary["final_parameters"]["trainable"]})
        tracker.finish(output)
        env.close()
        return summary
    except Exception as exc:
        summary.update(status="FAILED",error=type(exc).__name__+": "+str(exc),elapsed_seconds=time.perf_counter()-started)
        write_json(output/"summary.json",summary)
        write_json(output/"progress.json",summary)
        write_json(output/"resources.json",guard.samples)
        if model is not None and trainer is not None:
            try:
                save_checkpoint(output/"checkpoints/interrupted.pt",model,trainer,
                                metadata={"task_id":locals().get("current_task",task_id),"interrupted":True},
                                growth=locals().get("growth"))
            except Exception as save_error:
                write_json(output/"checkpoint_error.json",{"error":str(save_error)})
        if tracker:
            tracker.finish(output,"FAILED")
        raise
    finally:
        lock.close()
