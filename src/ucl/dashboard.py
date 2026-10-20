"""Local Streamlit interface. Replay uses saved frames; inference is a child process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from ucl.playback import (
    comparable_recordings,
    confined_path,
    discover_architecture_configs,
    discover_recordings,
    load_recording,
    next_playback_step,
    read_json,
    read_metrics,
    step_details,
    synchronized_indices,
)
from ucl.presentation import (
    architecture_description,
    architecture_name,
    compatible_checkpoints,
    object_name,
    panel_label,
    phase_label,
    run_catalog,
    study_catalog,
    study_checkpoint_label,
    study_manifest,
    study_task_label,
    study_task_summary,
    task_description,
    task_name,
)
from ucl.study_format import panel_order, resolve_study_run
from ucl.study_recording import MAX_MAPS, check_request, run_definition

LAB_ROOT = Path(os.environ.get("UCL_LAB_ROOT", Path(__file__).resolve().parents[2])).resolve()
ACCENT, MUTED = "#315dcc", "#58657a"


class EvaluationJobs:
    """One sequential evaluation queue shared by all dashboard sessions."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.job: dict | None = None
        self.worker: threading.Thread | None = None

    def status(self) -> dict | None:
        with self.lock:
            return None if self.job is None else dict(self.job)

    def start(
        self,
        config: Path,
        tasks: list[str],
        seed: int,
        checkpoint: Path | None = None,
        curriculum: str | None = None,
    ) -> dict:
        config = confined_path(config, self.root / "configs")
        selected_config = read_json(config)
        if selected_config.get("availability") == "unavailable":
            raise ValueError(selected_config.get("unavailable_reason", "Architecture unavailable"))
        if checkpoint:
            checkpoint = confined_path(checkpoint, self.root / "runs")
        if not tasks or any(not isinstance(task, str) or not task for task in tasks):
            raise ValueError("Select at least one task")
        selected = [curriculum] if curriculum else list(tasks)
        request = {
            "kind": "evaluate",
            "config": str(config),
            "tasks": list(tasks),
            "seed": int(seed),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "curriculum": curriculum,
        }

        def build(token):
            outputs = [self.root / "runs" / f"ui-{token}-{i:02d}" for i in range(len(selected))]
            commands = []
            for task, output in zip(selected, outputs, strict=True):
                command = [
                    sys.executable,
                    "-m",
                    "ucl.cli",
                    "evaluate",
                    "--config",
                    str(config),
                    "--curriculum" if curriculum else "--task",
                    task,
                    "--seed",
                    str(int(seed)),
                    "--output",
                    str(output),
                ]
                if checkpoint:
                    command.extend(["--checkpoint", str(checkpoint)])
                commands.append(command)
            return outputs, commands

        return self._launch(request, build)

    def start_study(
        self, runs: list[str], checkpoint: str, task_key: str, panel: str, maps: list[int]
    ) -> dict:
        """Record study episodes in a child process; one command per source run."""
        runs = list(dict.fromkeys(runs))
        if not runs or len(runs) > 2:
            raise ValueError("Record one run, or one run plus its comparison")
        definitions = [run_definition(self.root, resolve_study_run(self.root, r)[1]) for r in runs]
        if len({definition["protocol_hash"] for definition in definitions}) != 1:
            raise ValueError("Both runs must belong to the same study protocol")
        _, _, numbers = check_request(definitions[0], task_key, panel, maps)
        request = {
            "kind": "study",
            "runs": runs,
            "checkpoint": checkpoint,
            "task_key": task_key,
            "panel": panel,
            "maps": numbers,
        }

        def build(token):
            outputs = [self.root / "runs" / f"ui-{token}-study-{i:02d}" for i in range(len(runs))]
            commands = [
                [
                    sys.executable,
                    "-m",
                    "ucl.cli",
                    "study-record",
                    "--run",
                    run_id,
                    "--checkpoint",
                    checkpoint,
                    "--task",
                    task_key,
                    "--panel",
                    panel,
                    "--maps",
                    ",".join(str(number) for number in numbers),
                    "--output",
                    str(output),
                ]
                for run_id, output in zip(runs, outputs, strict=True)
            ]
            return outputs, commands

        return self._launch(request, build)

    def retry(self) -> dict:
        """Launch the failed request again with new output folders; never retries by itself."""
        status = self.status()
        if status is None or status["state"] != "failed":
            raise RuntimeError("There is no failed run to retry")
        request = status.get("request") or {}
        if request.get("kind") == "study":
            return self.start_study(
                request["runs"],
                request["checkpoint"],
                request["task_key"],
                request["panel"],
                request["maps"],
            )
        if request.get("kind") == "evaluate":
            checkpoint = request.get("checkpoint")
            return self.start(
                Path(request["config"]),
                request["tasks"],
                request["seed"],
                Path(checkpoint) if checkpoint else None,
                request.get("curriculum"),
            )
        raise RuntimeError("This run did not record how to repeat it")

    def _launch(self, request: dict, build) -> dict:
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("An evaluation is already running")
            token = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
            )
            directory = self.root / "runs" / "ui_jobs"
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / f"{token}.log"
            outputs, commands = build(token)
            self.job = {
                "id": token,
                "state": "running",
                "completed": 0,
                "total": len(commands),
                "outputs": [str(p) for p in outputs],
                "log": str(log),
                "commands": commands,
                "request": request,
                "pid": None,
                "returncode": None,
            }
            (directory / f"{token}.json").write_text(json.dumps(self.job, indent=2))
            self.worker = threading.Thread(target=self._execute, args=(commands, log), daemon=True)
            self.worker.start()
            return dict(self.job)

    def _execute(self, commands: list[list[str]], log: Path) -> None:
        environment = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            JAX_PLATFORM_NAME="cpu",
            PYTHONDONTWRITEBYTECODE="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        try:
            with log.open("ab", buffering=0) as handle:
                for index, command in enumerate(commands):
                    process = subprocess.Popen(
                        command,
                        cwd=self.root,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    with self.lock:
                        self.job.update(pid=process.pid, current=index + 1)
                    result = process.wait()
                    with self.lock:
                        self.job.update(returncode=result, pid=None)
                        if result != 0:
                            self.job["state"] = "failed"
                            break
                        self.job["completed"] = index + 1
                with self.lock:
                    if self.job["state"] != "failed":
                        self.job["state"] = "finished"
        except Exception as exc:
            with self.lock:
                self.job.update(state="failed", error=str(exc))
        finally:
            with self.lock:
                log.with_suffix(".json").write_text(json.dumps(self.job, indent=2))


@st.cache_resource
def jobs(root: str) -> EvaluationJobs:
    return EvaluationJobs(Path(root))


@st.cache_data(max_entries=4, show_spinner=False)
def cached_recording(path: str, root: str, modified_ns: int):
    return load_recording(path, root)


VIEWS = ["Estudio", "Ver un agente", "Probar un modelo", "Resultados"]
VIEW_LABELS = {
    "Estudio": "Estudios",
    "Ver un agente": "Simulación",
    "Probar un modelo": "Configurar evaluación",
    "Resultados": "Resultados",
}
QUERY_VIEWS = {
    "estudio": "Estudio",
    "simulacion": "Ver un agente",
    "evaluar": "Probar un modelo",
    "resultados": "Resultados",
}
PHASE_ORDER = ["pilot", "sequence", "single_map"]
SINGLE_MAP_PANEL_HELP = {
    "validation": "Mapas de validación del estudio; esta fase midió su curva en el mapa 1.",
    "test": "Mapas de test del estudio; esta fase no tiene tabla antes y después.",
    "known": "Primeros mapas de entrenamiento; esta fase solo entrenó en el mapa 1.",
    "train": "Todos los mapas de entrenamiento; esta fase solo entrenó en el mapa 1.",
}
PANEL_HELP = {
    "validation": "Los mapas de la curva de aprendizaje.",
    "test": "Los mismos mapas de la tabla de test.",
    "known": "Mapas que el agente ya vio al entrenar.",
    "train": "Todos los mapas de entrenamiento.",
}
SPLIT_DESCRIPTIONS = {
    "development": "tareas de desarrollo",
    "validation": "selección del modelo",
    "heldout": "tareas reservadas para evaluación",
}
ACTION_LABELS = [
    "Avanzar una casilla",
    "Girar a la derecha",
    "Girar a la izquierda",
    "Recoger el objeto de enfrente",
    "Soltar el objeto enfrente",
    "Abrir o cerrar la puerta de enfrente",
]


def keep_valid_selection(key, options, preferred=None):
    if options and st.session_state.get(key) not in options:
        st.session_state[key] = preferred if preferred in options else options[0]


def navigate(view, episode=None, compare=None, study_run=None):
    st.session_state.pending_view = view
    if episode:
        st.session_state.pending_episode = str(episode)
    if compare:
        st.session_state.pending_compare = str(compare)
    if study_run:
        st.session_state.pending_study_run = str(study_run)
    st.rerun()


def lower_first(text):
    """Sentence-case a label inside a sentence without lowering task letters (tarea C)."""
    return text[:1].lower() + text[1:]


def panel_title(number, title, description):
    st.markdown(
        f'<div class="panel-heading"><span>{escape(number)}</span>'
        f"<div><strong>{escape(title)}</strong><small>{escape(description)}</small></div></div>",
        unsafe_allow_html=True,
    )


def architecture_label(identifier, config=None):
    return f"{identifier} · {architecture_name(identifier, config)}"


def source_row(path, catalog):
    return next((row for row in catalog if row["path"].parent in Path(path).parents), None)


def choose_evaluation(root: Path, manager: EvaluationJobs) -> None:
    st.caption("Configura una evaluación nueva: el agente actúa con pesos fijos y guarda un run.")
    configs, unavailable = discover_architecture_configs(root)
    tasks_path = root / "configs/tasks.json"
    if not configs or not tasks_path.exists():
        st.info("Todavía no hay arquitecturas o tareas disponibles.")
        return
    manifest = read_json(tasks_path)
    tasks = manifest["tasks"]
    configs = sorted(configs, key=lambda p: (not p.stem.startswith("EXP-"), p.stem))
    preferred = next((p for p in configs if p.stem == "EXP-002"), configs[0])
    keep_valid_selection("architecture", configs, preferred)
    model_panel, scenario_panel = st.columns(2, gap="large")
    with model_panel, st.container(border=True, key="model_panel"):
        panel_title("01", "Modelo", "Arquitectura + checkpoint = agente que se evaluará")
        config = st.selectbox(
            "Arquitectura",
            configs,
            format_func=lambda p: architecture_label(read_json(p)["architecture_id"], read_json(p)),
            key="architecture",
        )
        selected_config = read_json(config)
        st.caption(architecture_description(selected_config["architecture_id"], selected_config))
        checkpoint_rows, checkpoint_errors = compatible_checkpoints(root, selected_config)
        cp_by_path = {row["path"]: row for row in checkpoint_rows}
        checkpoint_options = [*cp_by_path, None]
        keep_valid_selection("checkpoint", checkpoint_options)
        checkpoint = st.selectbox(
            "Checkpoint · pesos guardados",
            checkpoint_options,
            format_func=lambda p: (
                f"{p.name} · {cp_by_path[p]['label']}"
                if p
                else "Pesos originales del LLM (congelados)"
                if selected_config.get("architecture", {}).get("kind") == "frozen_llm_baseline"
                else "No aplica: política aleatoria"
                if selected_config["architecture_id"] == "EXP-001"
                else "Sin checkpoint · inicialización nueva"
            ),
            key="checkpoint",
            help="Sólo se ofrecen checkpoints compatibles con la arquitectura elegida.",
        )
        if checkpoint:
            cp = cp_by_path[checkpoint]
            st.caption(f"Run de origen: {cp['run_id']}")
            if cp["steps"] is not None:
                st.caption(f"Experiencia guardada: {cp['steps']:,} training steps.")
            with st.expander("Procedencia del checkpoint"):
                st.write(cp["description"])
                st.code(str(checkpoint.relative_to(root)))
        elif (
            selected_config["architecture_id"] != "EXP-001"
            and selected_config.get("architecture", {}).get("kind") != "frozen_llm_baseline"
        ):
            st.warning("Las partes entrenables usarán pesos nuevos, sin experiencia previa.")
        with st.expander("Estructura de la arquitectura"):
            st.json(selected_config.get("architecture", {}))
    with scenario_panel, st.container(border=True, key="scenario_panel"):
        panel_title("02", "Escenario", "Task + seed = condiciones del episodio")
        environment = st.selectbox(
            "Entorno",
            sorted({t.get("adapter", "xland") for t in tasks.values()}),
            format_func=lambda name: "XLand · mundo de casillas" if name == "xland" else name,
            key="environment",
        )
        mode = st.radio(
            "Protocolo",
            ["Una tarea", "Un currículo"],
            format_func=lambda item: "Task individual" if item == "Una tarea" else "Currículo",
            key="evaluation_mode",
            horizontal=True,
        )
        curriculum = None
        if mode == "Una tarea":
            task_ids = [k for k, t in tasks.items() if t.get("adapter", "xland") == environment]
            keep_valid_selection("task", task_ids, "heldout-navigation")
            task_id = st.selectbox("Task · objetivo y reglas", task_ids, key="task")
            selected_tasks = [task_id]
            st.caption(task_description(tasks[task_id]))
            split = tasks[task_id].get("split", "sin registrar")
            st.caption(f"Split: **{split}** · {SPLIT_DESCRIPTIONS.get(split, 'grupo de tareas')}")
        else:
            curricula = manifest.get("curricula", {})
            if not curricula:
                st.info("No hay currículos configurados.")
                return
            curriculum = st.selectbox(
                "Currículo · secuencia de tareas", list(curricula), key="curriculum"
            )
            selected_tasks = [
                stage["task_id"]
                for stage in curricula[curriculum].get("stages", [])
                if tasks[stage["task_id"]].get("adapter", "xland") == environment
            ]
            st.caption(" → ".join(selected_tasks))
            st.caption("Se evalúan en orden, sin entrenamiento entre tareas.")
        seed = st.number_input(
            "Seed · semilla de evaluación",
            min_value=0,
            max_value=2**32 - 1,
            value=240000,
            step=1,
            key="seed",
            help="Fija la inicialización aleatoria. La misma task y seed repiten el escenario.",
        )
    with st.container(border=True):
        panel_title("03", "Ejecución", "Evaluar → guardar episodios → calcular métricas")
        context, action = st.columns([2, 1], gap="large")
        with context:
            st.write(
                "**Modo:** evaluación · **Dispositivo:** "
                f"{selected_config.get('device', 'cpu').upper()}"
            )
            st.caption(
                "Los pesos permanecen fijos. El run guarda la grabación, las métricas y el "
                "estado del modelo."
            )
            if selected_config.get("device") == "cuda":
                st.caption("Esta arquitectura necesita la GPU.")
        with action:
            status = manager.status()
            running = status is not None and status["state"] == "running"
            if st.button(
                "Ejecutar evaluación",
                type="primary",
                width="stretch",
                disabled=running or not selected_tasks,
                key="launch_evaluation",
            ):
                try:
                    manager.start(config, selected_tasks, int(seed), checkpoint, curriculum)
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"No se pudo iniciar: {exc}")
            st.caption("Después: abrir el episodio en Simulación.")
    if unavailable or checkpoint_errors:
        with st.expander("Disponibilidad y diagnósticos"):
            for item in unavailable:
                st.write(f"{item['name']}: {item['reason']}")
            for error in checkpoint_errors:
                st.caption(error)


def recording_results(outputs):
    """'Memoria LSTM: 3 de 4 con éxito · Sin memoria (control): 2 de 4' — one part per agent."""
    parts = []
    for output in outputs:
        records = [read_json(path) for path in sorted((output / "episodes").rglob("episode.json"))]
        if records:
            wins = sum(bool(record.get("success")) for record in records)
            name = architecture_name(records[0].get("architecture_id"))
            parts.append(f"{name}: {wins} de {len(records)} con éxito")
    return " · ".join(parts) + "." if parts else ""


@st.fragment(run_every=1.0)
def job_status(manager: EvaluationJobs) -> None:
    status = manager.status()
    if status is None:
        return
    request = status.get("request") or {}
    outputs = [Path(output) for output in status.get("outputs", [])]
    episodes = [
        p for output in outputs for p in sorted((output / "episodes").rglob("episode.json"))
    ]
    output = outputs[0] if outputs else None
    if request.get("kind") == "study":
        st.caption(
            f"Última grabación: tarea {request['task_key']} · "
            f"{panel_label(request['panel'])} · "
            f"{lower_first(study_checkpoint_label(request['checkpoint']))}"
        )
    elif output and (output / "summary.json").exists():
        last_summary = read_json(output / "summary.json")
        st.caption(
            "Última prueba: "
            + architecture_name(last_summary.get("architecture_id"))
            + " · "
            + task_name(last_summary.get("metrics", {}).get("task_id", ""))
        )
    if status["state"] == "running":
        if request.get("kind") == "study":
            expected = len(request["maps"]) * len(request["runs"])
            st.info(
                f"Grabando episodios: {len(episodes)} de {expected}. "
                "Puedes cambiar de vista mientras termina."
            )
            st.progress(min(len(episodes) / expected, 1.0))
        else:
            st.info("El agente está haciendo la prueba. Puedes cambiar de vista mientras termina.")
            st.progress(status["completed"] / status["total"])
    elif status["state"] == "finished":
        if request.get("kind") == "study":
            st.success(f"{len(episodes)} episodios grabados. " + recording_results(outputs))
        else:
            st.success("Prueba terminada. Ya puedes ver qué hizo el agente.")
        if episodes and st.button("Abrir en Simulación", key="view_finished_job"):
            pair = [path for path in episodes if path.parent.name == episodes[0].parent.name]
            navigate("Ver un agente", pair[0], compare=pair[1] if len(pair) > 1 else None)
    else:
        st.error("La prueba no terminó. El motivo está en el registro técnico de abajo.")
        if st.button("Reintentar", key="retry_job", type="primary"):
            try:
                manager.retry()
                st.session_state.pop("retry_error", None)
                st.rerun()
            except (OSError, ValueError, RuntimeError) as exc:
                # Kept in the session: this fragment redraws every second.
                st.session_state.retry_error = (status["id"], str(exc))
        failed_retry = st.session_state.get("retry_error")
        if failed_retry and failed_retry[0] == status["id"]:
            st.error(f"No se pudo reintentar: {failed_retry[1]}")
    with st.expander("Registro técnico de esta prueba"):
        log = Path(status["log"])
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 10000))
                st.code(handle.read().decode(errors="replace"))
    if (
        status["state"] in {"finished", "failed"}
        and st.session_state.get("seen_job") != status["id"]
    ):
        st.session_state.seen_job = status["id"]
        st.rerun()


def inspect_results(root: Path) -> None:
    catalog, errors = run_catalog(root)
    if not catalog:
        st.info("Todavía no hay runs terminados. Crea uno en Configurar evaluación.")
        return
    selection, evidence = st.columns([1, 2.7], gap="large")
    with selection, st.container(border=True, key="selection_panel"):
        panel_title("01", "Ejecución", "Elige el run del que quieres ver evidencia")
        models = list(
            dict.fromkeys(row["summary"].get("architecture_id", "desconocido") for row in catalog)
        )
        keep_valid_selection("result_model", models, "EXP-002")
        model = st.selectbox(
            "Arquitectura", models, format_func=architecture_label, key="result_model"
        )
        candidates = [
            row for row in catalog if row["summary"].get("architecture_id", "desconocido") == model
        ]
        by_path = {row["path"]: row for row in candidates}
        keep_valid_selection("result", list(by_path))
        chosen = st.selectbox(
            "Run · ejecución guardada",
            list(by_path),
            format_func=lambda p: by_path[p]["run_id"],
            key="result",
        )
        summary = by_path[chosen]["summary"]
        metrics = summary.get("metrics", {})
        st.caption(
            f"Estado: {summary.get('status', 'sin registrar')} · "
            f"Modo: {summary.get('mode', 'sin registrar')}"
        )
        st.markdown("**Task evaluada**")
        st.code(metrics.get("task_id", "No registrada"), language=None)
        split = metrics.get("split", "sin registrar")
        st.caption(f"Split: {split} · {SPLIT_DESCRIPTIONS.get(split, 'grupo de tareas')}")
        st.caption("Seeds: " + ", ".join(str(seed) for seed in metrics.get("seeds", [])))
        if summary.get("checkpoint"):
            with st.expander("Checkpoint guardado por el run"):
                st.code(summary["checkpoint"], language=None)
                st.caption(
                    "Estado guardado de esta ejecución. El entrenamiento previo puede "
                    "proceder de otro run."
                )
        url = summary.get("mlflow", {}).get("url")
        if url:
            st.link_button("Abrir run en MLflow", url, width="stretch")
        recordings, _ = discover_recordings(root)
        episodes = [
            row["path"]
            for row in recordings
            if chosen.parent in Path(row["path"]).parents
            and row["task_id"] == metrics.get("task_id")
        ]
        if episodes and st.button("Ver episodio", key="result_episode", width="stretch"):
            navigate("Ver un agente", episodes[0])
    with evidence:
        with st.container(border=True, key="evaluation_panel"):
            panel_title("02", "Evaluación", "Rendimiento en la task y las seeds seleccionadas")
            rate, count = metrics.get("success_rate"), metrics.get("episodes")
            success_column, reward_column, count_column = st.columns(3)
            success_column.metric(
                "Success rate", f"{rate:.0%}" if isinstance(rate, (int, float)) else "—"
            )
            success_column.caption("Episodios que cumplieron la meta.")
            reward = metrics.get("reward_mean")
            reward_column.metric(
                "Reward medio", f"{reward:.3f}" if isinstance(reward, (int, float)) else "—"
            )
            reward_column.caption("Recompensa media por episodio.")
            count_column.metric("Episodios evaluados", count if count is not None else "—")
            count_column.caption("Tamaño de esta evaluación.")
            if rate == 0 and count:
                st.info("No resolvió la tarea en esta evaluación. Esto no demuestra aprendizaje.")
            elif count:
                st.caption(
                    "Para comparar arquitecturas: mismas tareas, experiencia y recursos; "
                    "varias repeticiones."
                )
            stages = summary.get("stages", [])
            if len(stages) > 1:
                st.caption(
                    "Arriba se muestra la última task evaluada; el currículo completo está aquí:"
                )
                with st.expander("Evaluación por etapa del currículo"):
                    st.dataframe(
                        [
                            {
                                "Task": row.get("task_id", ""),
                                "Episodios": row.get("episodes"),
                                "Success rate": row.get("success_rate"),
                                "Reward medio": row.get("reward_mean"),
                                "Training steps acumulados": row.get("training_steps_after"),
                            }
                            for row in stages
                        ],
                        hide_index=True,
                    )
        with st.container(border=True):
            panel_title(
                "03", "Entrenamiento", "Experiencia y actualización de pesos dentro de este run"
            )
            steps = summary.get("new_training_steps", summary.get("training_steps", 0))
            if summary.get("mode") == "evaluate":
                st.write("**Sólo evaluación · 0 pasos nuevos de entrenamiento**")
                st.caption(
                    "Los pesos se mantuvieron fijos. El checkpoint puede tener "
                    "entrenamiento previo."
                )
            else:
                st.metric("Training steps nuevos", f"{steps:,}")
                st.caption(
                    "Interacciones con el entorno durante el entrenamiento. No son epochs "
                    "ni episodios."
                )
            rows = read_metrics(chosen.parent / "metrics.jsonl")
            useful = [row for row in rows if isinstance(row.get("episode_success"), (int, float))]
            if useful:
                task_ids = list(dict.fromkeys(row.get("task_id", "tarea") for row in useful))
                selected_task = st.selectbox(
                    "Task de la curva de entrenamiento", task_ids, key="curve_task"
                )
                points = [
                    {"Training step": row["step"], "Success (%)": row["episode_success"] * 100}
                    for row in useful
                    if row.get("task_id", "tarea") == selected_task
                ]
                st.line_chart(points, x="Training step", y="Success (%)")
                st.caption("Éxitos durante la práctica; no son la evaluación del panel anterior.")
            retention = summary.get("retention", {})
            if retention:
                st.markdown("**Retención · rendimiento al volver a una tarea anterior**")
                if not retention.get("retention_defined"):
                    st.caption(
                        "No definida: no hubo éxito inicial. Cero antes y después no "
                        "demuestra memoria."
                    )
                else:
                    st.write(
                        f"Antes: {retention.get('after_learning', 0):.0%} · "
                        f"Después: {retention.get('after_other_tasks', 0):.0%}"
                    )
        with st.expander("Instrumentación: losses, parámetros, crecimiento y archivos"):
            numeric = sorted(
                {
                    key
                    for row in rows
                    for key, value in row.items()
                    if type(value) in (int, float)
                    and any(term in key for term in ("loss", "reward", "success"))
                }
            )
            if numeric:
                metric = st.selectbox("Métrica de entrenamiento", numeric, key="metric")
                points = [
                    {"Training step": row.get("step", i), metric: row[metric]}
                    for i, row in enumerate(rows)
                    if type(row.get(metric)) in (int, float)
                ]
                st.line_chart(points, x="Training step", y=metric)
                st.caption("Una loss menor no demuestra por sí sola que resuelva más tareas.")
            st.code(str(chosen.relative_to(root)))
            st.json(summary, expanded=False)
            if errors:
                st.write(errors)


def learning_curve(summary):
    """Validation success per evaluation, with the acquisition criterion and the random agent."""
    import altair as alt

    keys = summary.get("task_keys", [])
    curves = summary.get("curves", {})
    points = [
        {
            "Tarea": key,
            "Interacciones": point["total_step"],
            "Paso en la tarea": point["step"],
            "Éxito": round(point["success_rate"] * 100, 1),
        }
        for key in keys
        for point in curves.get(key, [])
    ]
    if not points:
        return None, points
    several = len(keys) > 1
    x = alt.X(
        "Interacciones:Q",
        title="Interacciones de entrenamiento",
        axis=alt.Axis(labelExpr="datum.value ? format(datum.value, '~s') : '0'", grid=False),
    )
    y = alt.Y(
        "Éxito:Q",
        title=(
            "Éxito repitiendo el mapa 1 (%)"
            if summary.get("phase") == "single_map"
            else "Éxito en validación (%)"
        ),
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(values=[0, 25, 50, 75, 100]),
    )
    base = alt.Chart(alt.Data(values=points))
    layers = [
        base.mark_line(color=ACCENT, strokeWidth=2).encode(x=x, y=y, detail="Tarea:N"),
        base.mark_circle(color=ACCENT, size=64, opacity=1).encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("Tarea:N"),
                alt.Tooltip("Interacciones:Q", format=","),
                alt.Tooltip("Éxito:Q", title="Éxito (%)", format=".0f"),
            ],
        ),
    ]
    acquisition = summary.get("acquisition") or {}
    rule = next((acquisition[key].get("rule") for key in keys if key in acquisition), None) or {}
    lines = []
    if isinstance(rule.get("threshold"), (int, float)):
        lines.append(
            {
                "y": rule["threshold"] * 100,
                "label": f"Criterio {rule['threshold']:.0%}",
                "kind": "a",
            }
        )
    for key in keys:
        chance = (summary.get("random_baselines", {}).get(key, {}).get("validation") or {}).get(
            "success_rate"
        )
        if isinstance(chance, (int, float)):
            suffix = f" · tarea {key}" if several else ""
            lines.append({"y": chance * 100, "label": f"Azar{suffix} ({chance:.0%})", "kind": "b"})
    if lines:
        guide = alt.Chart(alt.Data(values=lines))
        layers.append(
            guide.mark_rule(color=MUTED, strokeWidth=1).encode(
                y="y:Q",
                strokeDash=alt.StrokeDash(
                    "kind:N",
                    scale=alt.Scale(domain=["a", "b"], range=[[6, 4], [2, 3]]),
                    legend=None,
                ),
            )
        )
        layers.append(
            guide.mark_text(color=MUTED, align="left", dx=4, dy=-7, fontSize=11).encode(
                x=alt.value(0), y="y:Q", text="label:N"
            )
        )
    marks = []
    for key in keys:
        step = (acquisition.get(key) or {}).get("step")
        hit = next((p for p in points if p["Tarea"] == key and p["Paso en la tarea"] == step), None)
        if hit:
            marks.append(
                {"x": hit["Interacciones"], "label": f"Adquirida{' ' + key if several else ''}"}
            )
    for stage in summary.get("stages", [])[1:]:
        marks.append(
            {"x": stage.get("start_total_step", 0), "label": f"Empieza {stage['task_key']}"}
        )
    if marks:
        event = alt.Chart(alt.Data(values=marks))
        layers.append(event.mark_rule(color=MUTED, strokeWidth=1, opacity=0.7).encode(x="x:Q"))
        layers.append(
            event.mark_text(color="#172334", align="left", dx=4, fontSize=11).encode(
                x="x:Q", y=alt.value(10), text="label:N"
            )
        )
    return alt.layer(*layers).properties(height=270), points


def acquisition_rule_text(summary):
    """The study's own acquisition rule, worded for the caption under the curve."""
    acquisition = summary.get("acquisition") or {}
    rule = next((value.get("rule") for value in acquisition.values() if value.get("rule")), None)
    if not rule or not isinstance(rule.get("threshold"), (int, float)):
        return ""
    parts = [
        f"Se da por adquirida con {rule.get('consecutive', 1)} evaluaciones seguidas "
        f"en {rule['threshold']:.0%} o más"
    ]
    if isinstance(rule.get("min_gain"), (int, float)) and rule["min_gain"] > 0:
        parts.append(f"al menos {rule['min_gain'] * 100:.0f} puntos sobre el inicio y el azar")
    return " y ".join(parts) + "."


def study_matrix(summary, panel):
    matrix = summary.get("matrix") or {}
    measured = {key for row in matrix.values() for key in row.get(panel, {})}
    trained = [key for key in summary.get("task_keys", []) if key in measured]
    keys = trained + sorted(measured - set(trained))  # Trained tasks first, then transfer.
    table = []
    for name, row in matrix.items():
        entry = {"Momento": study_checkpoint_label(name)}
        for key in keys:
            value = row.get(panel, {}).get(key, {}).get("success_rate")
            entry[f"Tarea {key}"] = None if value is None else round(value * 100, 1)
        table.append(entry)
    if table and panel == "test":
        entry = {"Momento": "Agente al azar"}
        for key in keys:
            value = (summary.get("random_baselines", {}).get(key, {}).get("test") or {}).get(
                "success_rate"
            )
            entry[f"Tarea {key}"] = None if value is None else round(value * 100, 1)
        table.append(entry)
    return table, keys


def checkpoint_order(names, keys):
    after = [f"after_{key}" for key in reversed(keys) if f"after_{key}" in names]
    best = [f"best_{key}" for key in reversed(keys) if f"best_{key}" in names]
    rest = [name for name in names if name not in after + best]
    return after + best + rest


CHECKPOINT_HELP = {
    "initial": "Pesos con los que empezó: muestra al agente antes de aprender.",
    "after": "Pesos al terminar de entrenar esa tarea: son los que mide la tabla de test.",
    "best": "Pesos con mejor validación. Sirven para seleccionar, no se reportan como resultado.",
}


def matrix_panels(summary):
    """Panels the before/after matrix measured, usual ones first (e.g. test, known)."""
    names = []
    for row in (summary.get("matrix") or {}).values():
        for name, value in row.items() if isinstance(row, dict) else ():
            if isinstance(value, dict) and value and name not in names:
                names.append(name)
    preferred = [name for name in ("test", "known") if name in names]
    return preferred + [name for name in names if name not in preferred]


def same_run(context, row, ambiguous=frozenset()):
    """Recordings name their run by its path under runs/; older ones by a folder name that
    only counts when no other run has the same one."""
    source = context.get("source_run")
    return source == row["run_id"] or (source == row["folder"] and source not in ambiguous)


def episode_key(path):
    """Button key tied to the episode itself, so a list that grows never shifts a click."""
    return "study_episode_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]


def run_label(row):
    moment = datetime.fromtimestamp(row["mtime"], ZoneInfo("America/Mexico_City"))
    return f"{moment:%d/%m %H:%M} · {row['folder']}"


def show_study(root: Path, manager: EvaluationJobs) -> None:
    rows, errors = study_catalog(root)
    finished = [row for row in rows if row["status"] == "FINISHED"]
    if not finished:
        st.info(
            "Todavía no hay ejecuciones de estudios terminadas. Un estudio aparece aquí solo en "
            "cuanto guarda su resumen en `runs/` con el formato de docs/DASHBOARD.md. STUDY-001 "
            "se lanza desde WSL con `.venv/bin/python -m ucl.cli study-pilot`."
        )
        if errors:
            with st.expander("Diagnósticos"):
                for error in errors:
                    st.caption(error)
        return
    ambiguous = {
        name for name, count in Counter(row["folder"] for row in rows).items() if count > 1
    }
    requested = st.session_state.pop("pending_study_run", "")
    target, notice = None, None
    if requested:
        target = next((row for row in finished if row["run_id"] == requested), None)
        named = [row for row in finished if row["folder"] == requested]
        if target is None and len(named) == 1:
            target = named[0]
        elif target is None and named:
            notice = f"Varias ejecuciones se llaman {requested}; elige la que buscas abajo."
        elif target is None:
            other = next((r for r in rows if requested in (r["run_id"], r["folder"])), None)
            notice = (
                f"La ejecución {requested} todavía no terminó ({other['status']})."
                if other
                else f"No se encontró la ejecución {requested}."
            )
    if target is None and "study_phase" not in st.session_state:
        # First visit, or back from another view (Streamlit drops widget values while away).
        last = st.session_state.get("study_last_run")
        target = next((row for row in finished if row["run_id"] == last), finished[0])
    if target:
        st.session_state.update(
            study_id=target["study_id"],
            study_phase=target["phase"],
            study_task="-".join(target["task_keys"]),
            study_arch=target["architecture_id"],
            study_seed=target["init_seed"],
            study_run_id=target["run_id"],
        )
    if notice:
        st.warning(notice)

    with st.container(border=True, key="study_selection"):
        studies = sorted({row["study_id"] for row in finished})
        keep_valid_selection("study_id", studies)
        description = "fase + tarea + arquitectura + semilla = una ejecución en MLflow"
        panel_title(
            "01",
            "Ejecución",
            f"Estudio + {description}" if len(studies) > 1 else f"{studies[0]} · {description}",
        )
        widths = [1.3, 1.1, 1, 1.6, 1] if len(studies) > 1 else [1.1, 1, 1.6, 1]
        columns = st.columns(widths, gap="medium")
        if len(studies) > 1:
            with columns[0]:
                study_id = st.selectbox("Estudio", studies, key="study_id")
            columns = columns[1:]
        else:
            study_id = studies[0]
        phase_col, task_col, arch_col, seed_col = columns
        pool = [row for row in finished if row["study_id"] == study_id]
        phases = sorted(
            {row["phase"] for row in pool},
            key=lambda p: (PHASE_ORDER.index(p) if p in PHASE_ORDER else 99, str(p)),
        )
        keep_valid_selection("study_phase", phases)
        with phase_col:
            phase = st.selectbox("Fase", phases, format_func=phase_label, key="study_phase")
        pool = [row for row in pool if row["phase"] == phase]
        trained = sorted({"-".join(row["task_keys"]) for row in pool})
        keep_valid_selection("study_task", trained)
        with task_col:
            task_label = st.segmented_control(
                "Tarea entrenada",
                trained,
                format_func=lambda item: item.replace("-", " → "),
                key="study_task",
                required=True,
            )
        pool = [row for row in pool if "-".join(row["task_keys"]) == task_label]
        architectures = sorted({row["architecture_id"] for row in pool})
        keep_valid_selection("study_arch", architectures)
        with arch_col:
            arch = st.segmented_control(
                "Arquitectura",
                architectures,
                format_func=architecture_name,
                key="study_arch",
                required=True,
            )
        pool = [row for row in pool if row["architecture_id"] == arch]
        seeds = sorted({row["init_seed"] for row in pool}, key=str)
        keep_valid_selection("study_seed", seeds)
        with seed_col:
            seed = st.segmented_control("Semilla", seeds, key="study_seed", required=True)
        matches = {row["run_id"]: row for row in pool if row["init_seed"] == seed}
        if len(matches) > 1:
            keep_valid_selection("study_run_id", list(matches))
            row = matches[
                st.selectbox(
                    f"{len(matches)} ejecuciones con esta combinación",
                    list(matches),
                    format_func=lambda run_id: run_label(matches[run_id]),
                    key="study_run_id",
                )
            ]
        else:
            row = next(iter(matches.values()))
    st.session_state.study_last_run = row["run_id"]
    st.query_params["view"] = "estudio"
    st.query_params["run"] = row["run_id"]
    summary, keys = row["summary"], row["task_keys"]
    peer = next(
        (
            other
            for other in finished
            if other["run_id"] != row["run_id"]
            and other["study_id"] == study_id
            and other["phase"] == phase
            and other["task_keys"] == keys
            and other["init_seed"] == seed
            and other["architecture_id"] != arch
        ),
        None,
    )

    with st.container(border=True, key="study_evidence"):
        title_col, link_col = st.columns([3, 1], vertical_alignment="center")
        with title_col:
            panel_title(
                "02",
                f"{architecture_name(arch)} · tarea {' → '.join(keys)} · semilla {seed}",
                f"{architecture_description(arch)} Run {row['run_id']}.",
            )
            badges = [f":blue-badge[{phase_label(phase)}]"]
            for key in keys:
                step = row["acquired_step"].get(key)
                if step:
                    badges.append(f":green-badge[Tarea {key}: adquirida a {step:,} interacciones]")
                elif summary.get("acquisition"):
                    badges.append(
                        f":gray-badge[Tarea {key}: no adquirida en {summary.get('steps', 0):,}]"
                    )
            st.markdown(" ".join(badges))
        with link_col:
            if row["mlflow_url"]:
                st.link_button(
                    "Abrir en MLflow",
                    row["mlflow_url"],
                    width="stretch",
                    help="Parámetros, métricas y archivos de esta ejecución en MLflow.",
                )
        curve_col, table_col = st.columns([1.35, 1], gap="large")
        with curve_col:
            st.markdown("**Curva de aprendizaje**")
            chart, points = learning_curve(summary)
            if chart is None:
                st.caption("Esta ejecución no registró evaluaciones de validación.")
            else:
                st.altair_chart(chart, width="stretch")
                episodes = next(
                    (p.get("episodes") for c in summary.get("curves", {}).values() for p in c),
                    None,
                )
                measured = (
                    f"Éxito en {episodes or 'los'} episodios del mapa 1 de entrenamiento"
                    if phase == "single_map"
                    else f"Éxito en {episodes or 'los'} mapas de validación"
                )
                st.caption(
                    f"{measured} tras cada tramo de entrenamiento. "
                    + acquisition_rule_text(summary)
                )
        with table_col:
            st.markdown("**Éxito antes y después de entrenar**")
            measured = matrix_panels(summary)
            if measured:
                keep_valid_selection("study_matrix_panel", measured, "test")
                matrix_panel = st.segmented_control(
                    "Mapas evaluados",
                    measured,
                    format_func=panel_label,
                    key="study_matrix_panel",
                    required=True,
                )
                table, columns = study_matrix(summary, matrix_panel)
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        f"Tarea {key}": st.column_config.ProgressColumn(
                            f"Tarea {key}",
                            min_value=0,
                            max_value=100,
                            format="%.0f%%",
                            width="small",
                        )
                        for key in columns
                    },
                )
                others = [key for key in columns if key not in keys]
                st.caption(
                    f"Entrenó en {' → '.join(keys)}."
                    + (
                        f" Las columnas {', '.join(others)} miden transferencia sin entrenarlas."
                        if others
                        else ""
                    )
                )
            else:
                st.caption("Esta fase no evalúa paneles de mapas antes y después.")
        if points:
            with st.expander("Datos de la curva"):
                st.dataframe(points, hide_index=True, width="stretch")

    with st.container(border=True, key="study_watch"):
        panel_title(
            "03",
            "Ver al agente en el entorno",
            "Checkpoint + tarea + mapas = episodios grabados que reproduces en Simulación",
        )
        study_recorder(row, peer, manager)

    records, _ = discover_recordings(root)
    records.sort(key=lambda r: Path(r["path"]).stat().st_mtime_ns, reverse=True)
    mine = [r for r in records if same_run(r.get("study") or {}, row, ambiguous)]
    if mine:
        partners = {}
        for other in records:
            context = other.get("study") or {}
            if peer and same_run(context, peer, ambiguous):
                slot = ("task_key", "panel", "map_number", "checkpoint")
                partners.setdefault(tuple(context.get(name) for name in slot), other["path"])
        with st.container(border=True, key="study_recordings"):
            panel_title("04", "Episodios grabados", "Ábrelos en Simulación para ver cada paso")
            for record in mine[:12]:
                s = record["study"]
                text_col, result_col, button_col = st.columns(
                    [3, 1, 1], vertical_alignment="center"
                )
                moment = lower_first(study_checkpoint_label(s["checkpoint"]))
                text_col.write(
                    f"Tarea {s['task_key']} · {panel_label(s['panel'])} · "
                    f"mapa {s['map_number']} · {moment}"
                )
                result_col.markdown(
                    ":green-badge[Éxito]" if record["success"] else ":gray-badge[Sin éxito]"
                )
                partner = partners.get(
                    (s["task_key"], s["panel"], s["map_number"], s["checkpoint"])
                )
                if button_col.button(
                    "Ver con comparación" if partner else "Ver",
                    key=episode_key(record["path"]),
                    width="stretch",
                ):
                    navigate("Ver un agente", record["path"], compare=partner)
            if len(mine) > 12:
                st.caption(
                    f"Se muestran los 12 más recientes de {len(mine)}; el resto está en Simulación."
                )
    if errors:
        with st.expander("Diagnósticos"):
            for error in errors:
                st.caption(error)


def study_recorder(row, peer, manager: EvaluationJobs) -> None:
    """Checkpoint, task and maps to record; the work itself runs as a background job."""
    definition, keys = row["definition"], row["task_keys"]
    if definition is None:
        st.warning(
            "No se encontró en configs/ la definición de este estudio con el mismo protocolo, "
            "así que no se pueden grabar episodios. Lo que ya midió sigue siendo válido."
        )
        job_status(manager)
        return
    names = checkpoint_order(row["checkpoints"], keys)
    if not names:
        st.warning("Esta ejecución no conserva checkpoints que se puedan grabar.")
        job_status(manager)
        return
    manifest = study_manifest(definition)
    model_col, task_col = st.columns(2, gap="large")
    with model_col:
        keep_valid_selection("study_checkpoint", names, f"after_{keys[-1]}" if keys else None)
        checkpoint = st.selectbox(
            "Checkpoint · pesos guardados",
            names,
            format_func=study_checkpoint_label,
            key="study_checkpoint",
        )
        prefix = checkpoint.split("_")[0]
        st.caption(
            "Pesos al terminar de entrenar esa tarea."
            if prefix == "after" and not row["summary"].get("matrix")
            else CHECKPOINT_HELP.get(prefix, "")
        )
        compare = False
        if peer and peer["definition"] is not None and checkpoint in peer["checkpoints"]:
            compare = st.checkbox(
                f"Grabar también «{architecture_name(peer['architecture_id'])}» en los "
                "mismos mapas",
                value=True,
                key="study_compare",
                help="Misma tarea entrenada y misma semilla: podrás verlos lado a lado.",
            )
    with task_col:
        options = list(manifest)
        keep_valid_selection("study_watch_task", options, keys[-1] if keys else None)
        watch_task = st.segmented_control(
            "Tarea del episodio",
            options,
            format_func=lambda key: study_task_label(key, manifest[key]["task"]),
            key="study_watch_task",
            required=True,
        )
        for label, text in study_task_summary(manifest[watch_task]["task"]):
            st.markdown(f"**{label}:** {text}")
        if watch_task not in keys:
            st.caption("No entrenó en esta tarea: verás qué hace sin haberla practicado.")
    panels = panel_order(definition, watch_task)
    if not panels:
        st.warning("Esta tarea no tiene paneles de mapas que se puedan grabar.")
        job_status(manager)
        return
    sealed = [panel_label(name) for name in definition["sealed_panels"]]
    panel_col, maps_col = st.columns(2, gap="large")
    with panel_col:
        keep_valid_selection("study_panel", panels, "test")
        panel = st.segmented_control(
            "Mapas",
            panels,
            format_func=panel_label,
            key="study_panel",
            required=True,
        )
        helps = SINGLE_MAP_PANEL_HELP if row["phase"] == "single_map" else PANEL_HELP
        notes = [helps.get(panel, "")]
        if sealed:
            notes.append(f"Sellado hasta la evaluación final del estudio: {', '.join(sealed)}.")
        st.caption(" ".join(note for note in notes if note))
    with maps_col:
        size = manifest[watch_task]["panel_sizes"].get(panel, 0)
        numbers = list(range(1, size + 1))
        maps = st.multiselect(
            "Mapas a grabar",
            numbers,
            default=numbers[:4],
            format_func=lambda number: f"Mapa {number}",
            max_selections=MAX_MAPS,
            key=f"study_maps_{watch_task}_{panel}",
        )
    status = manager.status()
    running = status is not None and status["state"] == "running"
    action_col, note_col = st.columns([1, 2], gap="large", vertical_alignment="center")
    with action_col:
        if st.button(
            "Grabar episodios",
            type="primary",
            width="stretch",
            disabled=running or not maps,
            key="study_record",
        ):
            runs = [row["run_id"]] + ([peer["run_id"]] if compare else [])
            try:
                manager.start_study(runs, checkpoint, watch_task, panel, maps)
                st.rerun()
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"No se pudo iniciar: {exc}")
    with note_col:
        agents = 2 if compare else 1
        sampling = (
            "Pesos fijos. Esta fase midió su curva repitiendo el mapa 1 de entrenamiento "
            "con otro muestreo, así que el episodio puede diferir del de la curva."
            if row["phase"] == "single_map"
            else "Pesos fijos y la misma regla de muestreo que las evaluaciones del estudio."
        )
        st.caption(
            f"{len(maps)} mapas × {agents} {'agentes' if agents > 1 else 'agente'}. {sampling}"
        )
    job_status(manager)


def show_signals(signals: dict, prefix: str = "") -> None:
    for key, value in signals.items():
        label = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            show_signals(value, label)
        elif key == "module_count" and type(value) is int:
            st.metric("Cantidad de módulos", value)
        elif key == "active_mechanisms" and isinstance(value, list) and value:
            active = value[0] if isinstance(value[0], list) else value
            if active and all(type(flag) is bool for flag in active):
                st.caption("Módulos usados en este paso: 1 = activo, 0 = inactivo.")
                st.bar_chart(
                    [{"Módulo": str(i + 1), "Activo": int(flag)} for i, flag in enumerate(active)],
                    x="Módulo",
                    y="Activo",
                )
        elif key in {"state_norm", "cell_norm", "hidden_norm"}:
            numbers = value if isinstance(value, list) else [value]
            if numbers and isinstance(numbers[0], list):
                numbers = numbers[0]
            if numbers and all(type(number) in (int, float) for number in numbers):
                st.caption(f"Norma del estado · {label}")
                if len(numbers) == 1:
                    st.write(f"{numbers[0]:.4f}")
                else:
                    st.bar_chart(
                        [
                            {"Módulo": str(i + 1), "Valor": number}
                            for i, number in enumerate(numbers)
                        ],
                        x="Módulo",
                        y="Valor",
                    )


def episode_inspector(record, before):
    details = step_details(record, before)
    metadata = record.metadata
    st.markdown("**Estado del paso**")
    if details["transition"]:
        transition = details["transition"]
        action = transition.get("action")
        label = (
            ACTION_LABELS[action]
            if type(action) is int and 0 <= action < len(ACTION_LABELS)
            else str(action)
        )
        st.caption("Acción que produjo este estado")
        st.write(label)
        pocket = (transition.get("observer") or {}).get("pocket")
        if isinstance(pocket, list):
            held = object_name(pocket) or ("nada" if pocket == [0, 0] else str(pocket))
            st.caption("En la mano · dato del observador")
            st.write(held[0].upper() + held[1:])
        reward = transition.get("reward")
        st.metric("Reward del paso", f"{reward:.3f}" if isinstance(reward, (int, float)) else "—")
    else:
        st.caption("Estado inicial · todavía no hay una acción.")
    st.markdown("**Actividad neuronal**")
    if details["signals"]:
        show_signals(details["signals"])
    else:
        st.caption("Sin señales registradas en este paso.")
    st.caption("Las normas describen magnitud del estado; no miden cuánto recuerda.")
    with st.expander("Datos del episodio"):
        st.write(f"Seed: {metadata['seed']} · Reward total: {metadata['reward']:.3f}")
        st.json(details["signals"], expanded=False)
        st.code(str(record.path))


def seek_playback():
    """The seek slider is a manual input, never the animation's source of truth."""
    st.session_state.replay_position = st.session_state.playback_step
    st.session_state.playback_playing = False
    st.session_state.playback_clock = time.monotonic()


def playback_command(command, last):
    if command == "toggle":
        st.session_state.playback_playing = not st.session_state.playback_playing
        if st.session_state.playback_playing and st.session_state.replay_position >= last:
            st.session_state.replay_position = 0
    else:
        st.session_state.playback_playing = False
        st.session_state.replay_position = (
            min(st.session_state.replay_position + 1, last) if command == "step" else 0
        )
    st.session_state.playback_clock = time.monotonic()


def playback_controls(recordings) -> None:
    signature = tuple(str(record.path) for record in recordings)
    if st.session_state.get("playback_signature") != signature:
        st.session_state.update(
            playback_signature=signature,
            replay_position=0,
            playback_step=0,
            playback_playing=False,
            playback_clock=time.monotonic(),
        )
    st.session_state.setdefault("replay_position", 0)
    last = max(record.length for record in recordings) - 1
    buttons = st.columns([1.3, 1, 1])
    buttons[0].button(
        "Pausar" if st.session_state.playback_playing else "Reproducir",
        key="play_pause",
        type="primary",
        width="stretch",
        on_click=playback_command,
        args=("toggle", last),
    )
    buttons[1].button(
        "Un paso", key="one_step", width="stretch", on_click=playback_command, args=("step", last)
    )
    buttons[2].button(
        "Reiniciar",
        key="reset_playback",
        width="stretch",
        on_click=playback_command,
        args=("reset", last),
    )
    # Widget values stay stable during timer ticks. Only canonical replay_position advances.
    with st.container(key="replay_canvas"):
        replay_fragment(recordings)
    timeline, speed_column = st.columns([2.5, 1], vertical_alignment="bottom")
    with timeline:
        if last > 0:
            st.slider(
                "Saltar al paso · selección manual",
                0,
                last,
                key="playback_step",
                on_change=seek_playback,
            )
    with speed_column:
        st.select_slider(
            "Velocidad",
            options=[0.25, 0.5, 1.0, 2.0, 4.0],
            value=1.0,
            format_func=lambda x: f"{x:g}×",
            key="speed",
        )


@st.fragment(run_every=0.2)
def replay_fragment(recordings) -> None:
    last = max(record.length for record in recordings) - 1
    before = st.session_state.replay_position
    was_playing = st.session_state.playback_playing
    now = time.monotonic()
    speed = st.session_state.get("speed", 1.0)
    step, playing = next_playback_step(
        before, last, was_playing, now - st.session_state.playback_clock, speed
    )
    st.session_state.replay_position = step
    st.session_state.playback_playing = playing
    if step != before:
        st.session_state.playback_clock += (step - before) / (speed * 2.0)
    indices = synchronized_indices([record.length for record in recordings], step)
    if len(recordings) == 1:
        record = recordings[0]
        world, inspector = st.columns([1.4, 1], gap="medium")
        with world:
            st.image(record.frames[indices[0]], width="stretch")
            st.caption("▲ Agente · zona clara: su vista local.")
        with inspector:
            episode_inspector(record, step)
    else:
        for column, record, index in zip(
            st.columns(len(recordings)), recordings, indices, strict=True
        ):
            with column:
                st.markdown(f"**{architecture_label(record.metadata['architecture_id'])}**")
                st.image(record.frames[index], width="stretch")
                episode_inspector(record, step)
    st.progress(
        step / last if last else 1.0, text=f"Environment step {step}/{last} · una acción por paso"
    )
    for record in recordings:
        result = "Éxito" if record.metadata["success"] else "Sin éxito"
        st.caption(f"Resultado del episodio completo: {result} · {record.length - 1} acciones")
    if len(recordings) > 1:
        st.caption(
            "Mismo paso en ambos episodios. Al terminar, cada uno conserva su último estado."
        )
    if was_playing and not playing:
        st.rerun()  # Refresh the play button when animation reaches its end.


def show_recordings(root: Path) -> None:
    records, errors = discover_recordings(root)
    provisional = set()
    provenance = root / "audit/execution_provenance.json"
    if provenance.exists():
        provisional.add(
            str((root / read_json(provenance).get("provisional_run", "__none__")).resolve())
        )
    records = [
        record
        for record in records
        if not any(Path(p) in Path(record["path"]).parents for p in provisional)
    ]
    if not records:
        st.info("No hay episodios guardados. Crea uno en Configurar evaluación.")
        return
    records.sort(key=lambda record: Path(record["path"]).stat().st_mtime_ns, reverse=True)
    all_by_path = {record["path"]: record for record in records}
    pending = st.session_state.pop("pending_episode", None)
    pending_compare = st.session_state.pop("pending_compare", None)
    if pending in all_by_path:
        row = all_by_path[pending]
        st.session_state.update(
            watch_model=row["architecture_id"], watch_task=row["task_id"], episode_a=pending
        )
        if pending_compare in all_by_path:
            st.session_state.update(compare_agents=True, episode_b=pending_compare)
    catalog, catalog_errors = run_catalog(root)

    def record_label(path):
        record = all_by_path[path]
        moment = datetime.fromtimestamp(
            Path(path).stat().st_mtime, ZoneInfo("America/Mexico_City")
        ).strftime("%d/%m %H:%M")
        study = record.get("study")
        if study:
            result = "éxito" if record.get("success") else "sin éxito"
            return (
                f"{panel_label(study['panel'])} · mapa {study['map_number']}"
                f" · {lower_first(study_checkpoint_label(study['checkpoint']))} · semilla "
                f"{study.get('init_seed')} · {result} · {moment}"
            )
        source = source_row(path, catalog)
        run_name = source["run_id"] if source else Path(path).parents[2].name
        return f"{moment} · seed {record['seed']} · {run_name}"

    selection, workspace = st.columns([1, 2.7], gap="large")
    with selection, st.container(border=True, key="selection_panel"):
        panel_title("01", "Selección", "Qué episodio estás observando")
        models = list(dict.fromkeys(record["architecture_id"] for record in records))
        keep_valid_selection("watch_model", models, "EXP-002")
        model = st.selectbox(
            "Arquitectura", models, format_func=architecture_label, key="watch_model"
        )
        st.caption("Estructura de la red del agente.")
        task_ids = list(
            dict.fromkeys(
                record["task_id"] for record in records if record["architecture_id"] == model
            )
        )
        keep_valid_selection("watch_task", task_ids, "heldout-navigation")
        task = st.selectbox(
            "Task",
            task_ids,
            format_func=lambda identifier: f"{task_name(identifier)} · {identifier}",
            key="watch_task",
            help="Objetivo y reglas del entorno.",
        )
        matches = [
            record
            for record in records
            if record["architecture_id"] == model and record["task_id"] == task
        ]
        by_path = {record["path"]: record for record in matches}
        keep_valid_selection("episode_a", list(by_path))
        first = st.selectbox(
            "Episodio guardado", list(by_path), format_func=record_label, key="episode_a"
        )
        st.caption("Un episodio = un intento completo, desde el inicio hasta el final.")
        base = all_by_path[first]
        st.markdown(f"**Seed:** `{base['seed']}`")
        split = base.get("split") or "no registrado"
        st.caption(f"Split: **{split}** · {SPLIT_DESCRIPTIONS.get(split, 'grupo de tareas')}")
        compare = st.checkbox("Comparar arquitecturas", key="compare_agents")
        paths = [first]
        if compare:
            candidates = [
                record["path"]
                for record in records
                if record["path"] != first
                and record["architecture_id"] != model
                and comparable_recordings(record, base)
            ]
            if candidates:
                keep_valid_selection("episode_b", candidates)
                second = st.selectbox(
                    "Segundo episodio",
                    candidates,
                    format_func=lambda p: (
                        architecture_label(all_by_path[p]["architecture_id"])
                        + " · "
                        + record_label(p)
                    ),
                    key="episode_b",
                )
                paths.append(second)
                st.caption("Misma task y seed. El entrenamiento de ambos puede ser distinto.")
            else:
                st.info("No hay otra arquitectura con esta misma task y seed.")
    with workspace, st.container(border=True, key="simulator"):
        panel_title(
            "02",
            "Simulación",
            "REPLAY GRABADO · reproduce acciones; no ejecuta ni entrena el modelo",
        )
        try:
            recordings = [
                cached_recording(path, str(root), Path(path).stat().st_mtime_ns) for path in paths
            ]
            config = recordings[0].metadata.get("environment", {}).get("config", task)
            if recordings[0].metadata.get("study") and isinstance(config, dict):
                st.markdown(
                    " · ".join(f"**{label}:** {text}" for label, text in study_task_summary(config))
                )
            else:
                st.markdown(f"**Objetivo:** {task_description(config)}")
            playback_controls(recordings)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"No se puede abrir este episodio: {exc}")
    with st.expander("Contexto del episodio: arquitectura, run y archivos"):
        st.write(architecture_description(model))
        study = base.get("study")
        source = None if study else source_row(first, catalog)
        if study:
            st.write(
                f"**Ejecución de {study.get('study_id', 'estudio')}:** `{study['source_run']}` · "
                f"{study_checkpoint_label(study['checkpoint'])} · "
                f"{panel_label(study['panel'])}, mapa {study['map_number']}."
            )
            st.caption(
                "Pesos fijos y la regla de muestreo de los paneles del estudio. En la fase de un "
                "mapa, la curva usó otro muestreo: este episodio puede no ser uno de esa curva."
                if study.get("phase") == "single_map"
                else "Pesos fijos y la misma regla de muestreo que las evaluaciones del estudio."
            )
            links = st.columns(2)
            mlflow_url = (study.get("mlflow") or {}).get("url")
            if mlflow_url:
                links[0].link_button("Abrir en MLflow", mlflow_url, width="stretch")
            if links[1].button(
                "Ver su ejecución en Estudios", key="episode_study", width="stretch"
            ):
                navigate("Estudio", study_run=study["source_run"])
        if source:
            st.write(f"**Run:** `{source['run_id']}` · ejecución que generó este episodio.")
            st.caption(f"Modo del run: {source['summary'].get('mode', 'no registrado')}")
            if st.button("Ver métricas de este run", key="episode_results"):
                st.session_state.result_model = model
                st.session_state.result = source["path"]
                navigate("Resultados")
        st.caption(
            "El mapa completo y el objetivo son para el observador. El agente sólo recibe "
            "la vista permitida por el entorno."
        )
        if errors or catalog_errors:
            st.write(errors + catalog_errors)


def main() -> None:
    st.set_page_config(page_title="Universal Cognitive Lab", page_icon="🧠", layout="wide")
    st.markdown(
        """
    <style>
    .stApp {background: #f4f6fa; color: #172334;}
    .st-key-selection_panel, .st-key-model_panel, .st-key-scenario_panel, .st-key-simulator,
    .st-key-evaluation_panel, .st-key-training_panel, .st-key-study_selection,
    .st-key-study_evidence, .st-key-study_watch, .st-key-study_recordings {
        background: white; border: 1px solid #dce3ef !important; border-radius: 14px;
    }
    [data-testid="stWidgetLabel"] p, .st-key-lab_view label p {color:#172334;}
    [data-testid="stBaseButton-secondary"] {background:#fff; color:#172334; border-color:#dce3ef;}
    .st-key-simulator [data-testid="stImage"] {max-width: 280px;}
    [data-testid="stToolbar"] {display:none;}
    .block-container {padding: 1.3rem 2rem 2rem; max-width: 1400px;}
    header[data-testid="stHeader"] {background: transparent; height: 1rem;}
    h1 {font-size: 1.65rem !important; letter-spacing: -.035em; padding: 0 !important;}
    h2, h3 {font-size: 1.15rem !important;}
    [data-testid="stVerticalBlock"] {gap: .65rem;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 14px;}
    [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
        background: white;
        border-radius: 14px;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #58657a !important; font-size: .79rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    [data-testid="stImage"] img {image-rendering: pixelated; border-radius: 8px;}
    .panel-heading {display:flex; gap:10px; margin:0 0 8px; align-items:flex-start;}
    .panel-heading > span {font:600 11px monospace;
        color:#315dcc;
        background:#eaf0ff;
        padding:7px;
        border-radius:7px;}
    .panel-heading strong {display:block; font-size:1rem; line-height:1.35;}
    .panel-heading small {display:block;
        font-size:.72rem;
        color:#647187;
        margin-top:3px;
        line-height:1.4;}
    .lab-heading {display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:4px;}
    .lab-label {font:600 10px monospace; letter-spacing:.08em; color:#59718e;}
    div[role="radiogroup"] {gap:.45rem;}
    .st-key-lab_view div[role="radiogroup"] label {background:#e9edf4;
        border:1px solid #dce2ed;
        padding:8px 18px;
        border-radius:8px;}
    .st-key-lab_view div[role="radiogroup"] label:has(input:checked) {background:#e6edff;
        border-color:#849fdf;}
    .st-key-lab_view div[role="radiogroup"] label > div:first-child {display:none;}
    @media (max-width: 700px) {.block-container {padding:1rem;} .lab-label {display:none;}}
    </style>
    <div class="lab-heading"><h1>Universal Cognitive Lab</h1>
    <span class="lab-label">LOCAL / RESEARCH WORKSPACE</span></div>
    """,
        unsafe_allow_html=True,
    )
    st.caption("Arquitectura + checkpoint → agente · Task + seed → episodio · Run → métricas")
    if "query_applied" not in st.session_state:
        # Deep links (?view=estudio&run=<folder>) let an MLflow run point at its visual replay.
        st.session_state.query_applied = True
        requested = QUERY_VIEWS.get(st.query_params.get("view", ""))
        if st.query_params.get("run"):
            st.session_state.pending_study_run = st.query_params["run"]
            requested = requested or "Estudio"
        if requested:
            st.session_state.pending_view = requested
    if "pending_view" in st.session_state:
        st.session_state.lab_view = st.session_state.pop("pending_view")
    if "lab_view" not in st.session_state:
        # Open on the study when it has results; otherwise on the replay, as before.
        studied = any(row["status"] == "FINISHED" for row in study_catalog(LAB_ROOT)[0])
        st.session_state.lab_view = "Estudio" if studied else "Ver un agente"
    view = st.radio(
        "Vista del laboratorio",
        VIEWS,
        format_func=VIEW_LABELS.get,
        horizontal=True,
        key="lab_view",
        label_visibility="collapsed",
    )
    if view != "Estudio":
        st.query_params.clear()
        st.query_params["view"] = next(k for k, v in QUERY_VIEWS.items() if v == view)
    root = LAB_ROOT
    manager = jobs(str(root))
    if view == "Estudio":
        show_study(root, manager)
    elif view == "Ver un agente":
        show_recordings(root)
    elif view == "Probar un modelo":
        choose_evaluation(root, manager)
        job_status(manager)
    else:
        inspect_results(root)
    with st.expander("Guía de conceptos y alcance de estas pruebas"):
        st.write(
            "**Arquitectura:** estructura de la red. **Checkpoint:** sus pesos guardados. "
            "**Task:** objetivo y reglas. **Seed:** inicialización aleatoria."
        )
        st.write(
            "**Run:** una ejecución del experimento. **Episodio:** un intento dentro del "
            "run. **Environment step:** una acción. **Training:** actualizar pesos con "
            "experiencia."
        )
        st.caption(
            "Las pruebas actuales verifican el funcionamiento del laboratorio. Todavía no "
            "demuestran una mejora científica."
        )


if __name__ == "__main__":
    main()
