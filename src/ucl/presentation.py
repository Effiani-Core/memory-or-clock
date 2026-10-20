"""Human-facing names and safe saved-model menus. Never load neural checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ARCHITECTURES = {
    "EXP-001": ("Agente al azar", "Elige acciones al azar. Sirve como referencia y no aprende."),
    "EXP-002": (
        "Memoria LSTM",
        "Una red con memoria aprende a decidir qué hacer. No usa un modelo de lenguaje.",
    ),
    "EXP-003": (
        "Memoria GRU",
        "Otra red con memoria, más simple que LSTM. No usa un modelo de lenguaje.",
    ),
    "EXP-004": (
        "Memoria modular RIM",
        "Varios módulos de memoria; sólo algunos se actualizan en cada paso.",
    ),
    "EXP-005": (
        "RIM que puede crecer",
        "Puede añadir módulos durante el entrenamiento, dentro de un límite.",
    ),
    "EXP-006": (
        "Modelo de lenguaje",
        "Un modelo de lenguaje local elige las acciones. Sus pesos permanecen fijos.",
    ),
    "EXP-007": (
        "Modelo de lenguaje + RIM",
        "Usa un modelo de lenguaje fijo y entrena una memoria modular.",
    ),
    "EXP-008": (
        "Modelo de lenguaje + RIM que crece",
        "Usa un modelo de lenguaje fijo y una memoria que puede añadir módulos.",
    ),
    "CONTROL-GROWTH-NONE": (
        "Control: RIM sin crecer",
        "Usa la red expandible con el crecimiento desactivado.",
    ),
    "CONTROL-GROWTH-PLATEAU": (
        "Control: RIM crece al estancarse",
        "Prueba añadir módulos cuando la recompensa reciente deja de mejorar.",
    ),
    "CONTROL-RIM-MAX": (
        "Control: RIM grande desde el inicio",
        "Empieza con la capacidad máxima para comparar el efecto de añadir módulos después.",
    ),
    "CUSTOM-EXAMPLE": (
        "Componente personalizado",
        "Ejemplo que comprueba que podemos conectar una arquitectura propia.",
    ),
    "OPTIONAL-LARGER-LLM": (
        "Modelo de lenguaje más grande",
        "Alternativa opcional. Sólo se puede ejecutar si está instalado "
        "y cabe en la memoria disponible.",
    ),
    "S001-LSTM": (
        "Memoria LSTM",
        "Encoder + LSTM: puede recordar lo que vio en los pasos anteriores del episodio.",
    ),
    "S001-MLP": (
        "Sin memoria (control)",
        "El mismo encoder sin parte recurrente: decide sólo con lo que ve en este paso.",
    ),
}
STUDY_TASK_ID = re.compile(r"^study(\d{3})-([A-Z])-(.+)$")
STUDY_PHASES = {
    "pilot": "Piloto",
    "single_map": "Prueba de un mapa",
    "sequence": "Secuencia",
}
STUDY_PANELS = {
    "validation": "Validación",
    "test": "Test",
    "known": "Mapas ya vistos",
    "train": "Entrenamiento",
    "final": "Final",
}
TASKS = {
    "navigation": ("Llegar al objetivo", "Moverse hasta la casilla objetivo."),
    "pickup": ("Recoger un objeto", "Recoger y conservar el objeto que pide la meta."),
    "hold_transform": (
        "Transformar al recoger",
        "Recoger un objeto puede transformarlo; conseguir el resultado pedido.",
    ),
    "adjacency_transform": (
        "Transformar al juntar objetos",
        "Colocar objetos junto a otros puede transformarlos.",
    ),
    "changed_rule": (
        "Adaptarse a una regla distinta",
        "Resolver una tarea donde cambia la regla de transformación.",
    ),
    "composition": (
        "Combinar transformaciones",
        "Usar varias transformaciones para llegar a la meta.",
    ),
    "directional": (
        "Colocar en una dirección concreta",
        "La transformación depende de la dirección en que se colocan los objetos.",
    ),
    "directional_adjacency": (
        "Colocar en una dirección concreta",
        "La transformación depende de la dirección en que se colocan los objetos.",
    ),
}
SPLITS = {
    "development": "Práctica",
    "validation": "Selección del modelo",
    "heldout": "Prueba final",
}
COLORS = {
    1: "rojo",
    2: "verde",
    3: "azul",
    4: "morado",
    5: "amarillo",
    6: "gris",
    7: "negro",
    8: "naranja",
    9: "blanco",
    10: "café",
    11: "rosa",
}
TILES = {
    3: "la pelota",
    4: "el cuadrado",
    5: "la pirámide",
    6: "la casilla meta",
    7: "la llave",
    11: "el hexágono",
    12: "la estrella",
}
CHECKPOINTS = {
    "last": ("Último estado guardado", "Último estado guardado de esta ejecución."),
    "best": (
        "Seleccionado por validación",
        "Elegido según la validación registrada. Puede coincidir con el inicial; "
        "no garantiza aprendizaje.",
    ),
    "initial": (
        "Estado al iniciar la prueba",
        "Estado con el que empezó esta ejecución. Puede contener pesos aprendidos "
        "en una prueba anterior.",
    ),
}


def _human(value):
    return str(value or "Sin identificar").replace("_", " ").replace("-", " ").strip().capitalize()


def _architecture_key(identifier):
    value = str(identifier or "")
    for suffix, label in (
        ("-fresh_same_initialization", "sin experiencia previa"),
        ("-experienced", "con experiencia previa"),
    ):
        if value.endswith(suffix):
            return value[: -len(suffix)], label
    return value, None


def architecture_name(identifier, config=None):
    key, suffix = _architecture_key(identifier)
    name = ARCHITECTURES.get(key, (_human((config or {}).get("name", identifier)), ""))[0]
    return f"{name} · {suffix}" if suffix else name


def architecture_description(identifier, config=None):
    key, _ = _architecture_key(identifier)
    return ARCHITECTURES.get(
        key, ("", "Arquitectura definida en la configuración de esta prueba.")
    )[1]


def split_name(split):
    return SPLITS.get(split, _human(split))


def _task_parts(task_or_id, split=None):
    if isinstance(task_or_id, dict):
        identifier = task_or_id.get("id", task_or_id.get("task_id", ""))
        family = task_or_id.get("family", task_or_id.get("task_family"))
        split = split or task_or_id.get("split")
    else:
        identifier, family = str(task_or_id or ""), None
    for prefix in SPLITS:
        if str(identifier).startswith(prefix + "-"):
            split = split or prefix
            family = family or str(identifier)[len(prefix) + 1 :]
            break
    return family or identifier, split


def task_name(task_or_id, split=None):
    identifier = task_or_id.get("id", "") if isinstance(task_or_id, dict) else str(task_or_id or "")
    study = STUDY_TASK_ID.match(identifier)
    if study:
        return f"{TASKS.get(study[3], (_human(study[3]), ''))[0]} · tarea {study[2]}"
    family, selected_split = _task_parts(task_or_id, split)
    name = TASKS.get(family, (_human(family), ""))[0]
    return f"{name} · {split_name(selected_split)}" if selected_split else name


def task_description(task_or_id):
    family, _ = _task_parts(task_or_id)
    generic = TASKS.get(family, ("", "Completar el objetivo de este escenario."))[1]
    if not isinstance(task_or_id, dict):
        return generic
    goal = task_or_id.get("goal", {})
    tile = goal.get("tile")
    if not isinstance(tile, (list, tuple)) or len(tile) != 2:
        return generic
    tile_name, color = TILES.get(tile[0]), COLORS.get(tile[1])
    if tile_name is None or color is None:
        return generic
    if goal.get("type") == "AgentOnTileGoal":
        exact = f"Llega a la casilla de color {color}."
    elif goal.get("type") == "AgentHoldGoal":
        exact = f"Recoge y conserva {tile_name} de color {color}."
    else:
        return generic
    return f"{exact} {generic}" if family not in ("navigation", "pickup") else exact


def object_name(tile):
    """'la pelota de color rojo' for [3, 1]; None when the tile is not a named object."""
    if not isinstance(tile, (list, tuple)) or len(tile) != 2:
        return None
    name, color = TILES.get(tile[0]), COLORS.get(tile[1])
    return f"{name} de color {color}" if name and color else None


def _rule_text(rule):
    """Plain wording of one XLand rule, or None when the rule type is not understood."""
    kind = str(rule.get("type", ""))
    product = object_name(rule.get("prod_tile"))
    if kind == "EmptyRule":
        return ""
    if kind == "AgentHoldRule" and object_name(rule.get("tile")) and product:
        return f"Al recoger {object_name(rule['tile'])}, se convierte en {product}."
    if kind.startswith("AgentNear") and object_name(rule.get("tile")) and product:
        return f"Al acercarse a {object_name(rule['tile'])}, se convierte en {product}."
    if kind.startswith("TileNear") and product:
        first, second = object_name(rule.get("tile_a")), object_name(rule.get("tile_b"))
        if first and second:
            return f"Al juntar {first} con {second}, se convierten en {product}."
    return None


def study_task_summary(task):
    """Observer-facing reading of a task's goal, rules and distractors as (label, text) pairs."""
    goal, rules = task.get("goal") or {}, task.get("rules") or []
    target = goal.get("tile")
    kind = goal.get("type")
    lines = []
    if kind == "AgentOnTileGoal" and object_name(target):
        lines.append(("Meta", f"Llegar a {object_name(target)}."))
    elif kind == "AgentHoldGoal" and object_name(target):
        lines.append(("Meta", f"Tener en la mano {object_name(target)}."))
    elif kind == "AgentNearGoal" and object_name(target):
        lines.append(("Meta", f"Quedar junto a {object_name(target)}."))
    else:
        lines.append(("Meta", task_description(task)))
    # Every tile a goal or rule names is part of the task, never a distractor.
    used = [list(value) for value in (target, goal.get("tile_a"), goal.get("tile_b")) if value]
    understood = kind in {"AgentOnTileGoal", "AgentHoldGoal", "AgentNearGoal"}
    for rule in rules:
        used += [
            list(rule[name])
            for name in ("tile", "tile_a", "tile_b", "prod_tile")
            if isinstance(rule.get(name), list)
        ]
        text = _rule_text(rule)
        if text is None:
            understood = False
            lines.append(("Regla", f"Transformación {_human(rule.get('type'))}."))
        elif text:
            lines.append(("Regla", text))
            source, product = rule.get("tile"), rule.get("prod_tile")
            if rule.get("type") == "AgentHoldRule" and list(product) == list(target or []):
                lines.append(("En la práctica", f"Recoger {object_name(source)}."))
    distractors = [
        object_name(tile)
        for tile in task.get("init_tiles", [])
        if list(tile) not in used and object_name(tile)
    ]
    if understood and distractors:
        lines.append(("Distractor", " · ".join(distractors) + " (no sirve para la meta)."))
    return lines


def study_checkpoint_label(name):
    if name == "initial":
        return "Al empezar · sin entrenar"
    if name.startswith("after_"):
        return f"Al terminar la tarea {name[len('after_') :]}"
    if name.startswith("best_"):
        return f"Mejor en validación · tarea {name[len('best_') :]}"
    return _human(name)


def phase_label(name):
    return STUDY_PHASES.get(name, _human(name))


def panel_label(name):
    return STUDY_PANELS.get(name, _human(name))


def study_task_label(key, task):
    """'B · Recoger un objeto' from the task family; unknown families keep their own name."""
    family = task.get("family") if isinstance(task, dict) else None
    return f"{key} · {TASKS.get(family, (_human(family),))[0]}" if family else str(key)


def study_manifest(definition):
    """Tasks of one study definition by key, with the size of every panel."""
    return {
        key: {
            "task_id": task_id,
            "task": definition["tasks"][task_id],
            "panel_sizes": {
                name: len(seeds) for name, seeds in definition["panels"][task_id].items()
            },
        }
        for key, task_id in definition["task_ids"].items()
    }


def study_catalog(root):
    """Return (study run rows, errors), newest first, for every study found by its format.

    Reads JSON only and never opens checkpoints. Each row carries the matching definition from
    ``configs/`` (same study and protocol hash) or ``None`` when the frozen files changed.
    """
    from .study_format import (
        definition_for,
        discover_studies,
        run_id_of,
        study_run_files,
    )

    root = Path(root).resolve()
    definitions, errors = discover_studies(root)
    rows = []
    try:
        paths = study_run_files(root)
    except (OSError, ValueError) as exc:
        return [], [*errors, f"No se pudieron leer las ejecuciones de estudios: {exc}"]
    for path in paths:
        try:
            summary = _read_json(path)
            if not summary.get("study_id"):
                continue
            if not summary.get("architecture_id") or not summary.get("protocol_hash"):
                raise ValueError("Falta la arquitectura o el hash del protocolo")
            for field in ("study_id", "architecture_id", "protocol_hash"):
                if not isinstance(summary[field], str):
                    raise ValueError(f"{field} debe ser texto")
            if not isinstance(summary.get("phase"), (str, type(None))):
                raise ValueError("phase debe ser texto")
            seed = summary.get("init_seed")
            if isinstance(seed, bool) or not isinstance(seed, (int, str, type(None))):
                raise ValueError("init_seed debe ser un número o texto")
            if not isinstance(summary.get("task_keys", []), list):
                raise ValueError("task_keys debe ser una lista")
            keys = [key for key in summary.get("task_keys", []) if isinstance(key, str)]
            acquisition = summary.get("acquisition") or {}
            matrix = summary.get("matrix") or {}
            last = keys[-1] if keys else None
            final = (matrix.get(f"after_{last}") or {}).get("test", {}).get(last, {})
            available = [
                name
                for name, entry in (summary.get("checkpoints") or {}).items()
                if isinstance(entry, dict)
                and (path.parent / str(entry.get("path", ""))).resolve().is_file()
                and (path.parent / str(entry.get("path", "")))
                .resolve()
                .is_relative_to(path.parent.resolve())
            ]
            rows.append(
                {
                    "path": path,
                    "run_id": run_id_of(root, path.parent),
                    "folder": path.parent.name,
                    "summary": summary,
                    "study_id": summary["study_id"],
                    "definition": definition_for(summary, definitions),
                    "architecture_id": summary["architecture_id"],
                    "status": summary.get("status"),
                    "phase": summary.get("phase"),
                    "task_keys": keys,
                    "init_seed": summary.get("init_seed"),
                    "acquired_step": {
                        key: (acquisition.get(key) or {}).get("step") for key in keys
                    },
                    "test_after": final.get("success_rate"),
                    "mlflow_url": (summary.get("mlflow") or {}).get("url"),
                    "checkpoints": available,
                    "mtime": path.stat().st_mtime,
                }
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"No se pudo leer una ejecución de estudio ({path.parent.name}): {exc}")
    rows.sort(key=lambda row: (row["mtime"], row["run_id"]), reverse=True)
    return rows, errors


def architecture_fingerprint(config):
    architecture = config.get("architecture", config)
    return hashlib.sha256(
        json.dumps(architecture, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _compatible_architecture(selected, recorded):
    """Mirror strict selection semantics without importing torch or opening .pt."""
    left, right = deepcopy(selected), deepcopy(recorded)
    if not isinstance(left, dict) or not isinstance(right, dict) or not left or not right:
        return False
    if left.get("kind", "graph") != right.get("kind", "graph"):
        return False
    if left.get("kind", "graph") == "graph":
        a_nodes, b_nodes = left.get("nodes"), right.get("nodes")
        if (
            not isinstance(a_nodes, list)
            or not isinstance(b_nodes, list)
            or len(a_nodes) != len(b_nodes)
        ):
            return False
        for a, b in zip(a_nodes, b_nodes):
            if not isinstance(a, dict) or not isinstance(b, dict):
                return False
            if a.get("type") == b.get("type") == "expandable_rim":
                requested, actual = a.get("params", {}), b.get("params", {})
                minimum, maximum, count = (
                    requested.get("modules", 2),
                    requested.get("max_modules", 8),
                    actual.get("modules", 2),
                )
                if (
                    any(type(value) is not int for value in (minimum, maximum, count))
                    or not minimum <= count <= maximum
                ):
                    return False
                requested.pop("modules", None)
                actual.pop("modules", None)
    return left == right


def _read_json(path):
    if path.stat().st_size > 8 * 2**20:
        raise ValueError("El registro supera el tamaño permitido")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("El registro no es un objeto JSON")
    return value


def _provisional_paths(root, errors):
    path = root / "audit/execution_provenance.json"
    if not path.exists():
        return set()
    try:
        evidence = _read_json(path)
        values = evidence.get("provisional_runs", [])
        if evidence.get("provisional_run"):
            values = [*values, evidence["provisional_run"]]
        excluded = set()
        for value in values:
            candidate = (root / value).resolve()
            if not candidate.is_relative_to(root / "runs"):
                raise ValueError("La exclusión provisional sale del laboratorio")
            excluded.add(candidate)
        return excluded
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"No se pudo leer la revisión de ejecuciones: {exc}")
        # Known diagnostic is hidden conservatively if the audit file is damaged.
        return {(root / "runs/progressive-smoke").resolve()}


def run_catalog(root):
    """Return (original run rows, errors), newest first; never traverse MLflow mirrors."""
    from .playback import run_files

    root = Path(root).resolve()
    errors, rows = [], []
    provisional = _provisional_paths(root, errors)
    try:
        paths = run_files(root, "**/summary.json")
    except (OSError, ValueError) as exc:
        return [], [f"No se pudieron leer las pruebas: {exc}"]
    for path in paths:
        if any(
            path.parent == directory or path.parent.is_relative_to(directory)
            for directory in provisional
        ):
            continue
        try:
            summary = _read_json(path)
            if summary.get("study_id"):
                continue  # Study runs follow their own protocol; study_catalog lists them.
            config = summary.get("config", {})
            if (
                not isinstance(config, dict)
                or not summary.get("architecture_id")
                or not isinstance(summary.get("status"), str)
            ):
                raise ValueError("Faltan el modelo, el estado o la configuración")
            metrics = summary.get("metrics", {})
            task = metrics.get("task_id", config.get("task_id", ""))
            mtime = path.stat().st_mtime
            moment = datetime.fromtimestamp(mtime, ZoneInfo("America/Mexico_City")).strftime(
                "%d/%m %H:%M:%S"
            )
            label = (
                f"{architecture_name(summary['architecture_id'], config)} · {task_name(task)} · "
                f"semilla {config.get('seed', '?')} · {moment}"
            )
            rows.append(
                {
                    "path": path,
                    "summary": summary,
                    "label": label,
                    "run_id": str(path.parent.relative_to(root / "runs")),
                    "mtime": mtime,
                    "architecture_id": summary["architecture_id"],
                    "status": summary["status"],
                    "task_id": task,
                    "seed": config.get("seed"),
                }
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"No se pudo leer una prueba guardada ({path.parent.name}): {exc}")
    rows.sort(key=lambda row: (row["mtime"], row["run_id"]), reverse=True)
    return rows, errors


def compatible_checkpoints(root, config, runs=None):
    """Return compatible saved states with human labels. Runtime still validates .pt strictly.

    Compatibility here is a metadata filter, not a claim that an unread tensor file
    is valid. Different experiment/control IDs stay separate when an ID is selected.
    """
    root = Path(root).resolve()
    if runs is None:
        rows, errors = run_catalog(root)
    else:
        rows, errors = list(runs), []
    selected = config.get("architecture", config)
    selected_id = config.get("architecture_id")
    if not isinstance(selected, dict) or selected.get("kind") == "random":
        return [], errors
    candidates = []
    for row in rows:
        summary = row["summary"]
        if summary.get("status") != "FINISHED":
            continue
        if selected_id is not None and summary.get("architecture_id") != selected_id:
            continue
        directory = Path(row["path"]).parent.resolve()
        if not directory.is_relative_to(root / "runs"):
            errors.append("Se omitió una prueba fuera del laboratorio")
            continue
        try:
            recorded_config = summary.get("config", {})
            configuration_path = directory / "config.json"
            if configuration_path.exists():
                file_config = _read_json(configuration_path)
                if recorded_config.get("architecture") != file_config.get("architecture"):
                    raise ValueError("La configuración guardada no coincide con el resumen")
                if file_config.get("architecture_id") != summary.get("architecture_id"):
                    raise ValueError("El identificador del modelo no coincide con su configuración")
                recorded_config = file_config
            recorded = recorded_config.get("architecture")
            if not _compatible_architecture(selected, recorded):
                continue
            for kind, (name, description) in CHECKPOINTS.items():
                path = directory / "checkpoints" / f"{kind}.pt"
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(directory) or path.stat().st_size == 0:
                    raise ValueError("El modelo guardado está vacío o apunta fuera de su prueba")
                total = summary.get("training_steps")
                steps = total if kind == "last" and summary.get("mode") == "train" else None
                saved_at = datetime.fromtimestamp(
                    row["mtime"], ZoneInfo("America/Mexico_City")
                ).strftime("%d/%m %H:%M:%S")
                candidates.append(
                    {
                        "path": resolved,
                        "label": (
                            f"{name} · "
                            f"{str(steps) + ' pasos' if steps is not None else 'estado guardado'}"
                            f" · {saved_at}"
                        ),
                        "kind": kind,
                        "description": description + f" Origen: {row['label']}.",
                        "summary": summary,
                        "steps": steps,
                        "run_training_steps": total,
                        "run_id": row["run_id"],
                        "architecture_id": summary["architecture_id"],
                        "mtime": row["mtime"],
                        "architecture_fingerprint": architecture_fingerprint(recorded),
                        "compatibility_scope": "metadata; strict tensor validation on load",
                    }
                )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(
                f"No se pueden ofrecer los modelos de una prueba ({directory.name}): {exc}"
            )
    order = {"last": 0, "best": 1, "initial": 2}
    candidates.sort(key=lambda row: (-row["mtime"], order[row["kind"]], row["run_id"]))
    return candidates, errors
