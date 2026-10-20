# Universal Cognitive Lab

Laboratorio local para construir, entrenar y comparar arquitecturas neuronales. El entorno, aprendizaje, evaluación, grabación y seguimiento usan las mismas interfaces al cambiar la arquitectura.

**Las pruebas breves verifican la infraestructura. No demuestran AGI, novedad científica ni superioridad de un modelo. Estos smoke tests no constituyen por sí solos un resultado publicable.** El estado comprobado y los límites están en [el informe](IMPLEMENTATION_REPORT.md).

## Abrirlo

Desde PowerShell:

```powershell
wsl -d AIResearchLab -u researcher
```

Dentro de WSL:

```bash
cd memory-or-clock
.venv/bin/python -m ucl.cli dashboard
```

Abre [la interfaz](http://127.0.0.1:8501). **Simulación** separa la selección del episodio, el mapa y el inspector del paso. **Configurar evaluación** organiza arquitectura/checkpoint, escenario y ejecución. **Resultados** distingue métricas de evaluación y entrenamiento. Cada término conserva su nombre técnico con una explicación junto al control.

El entrenamiento también funciona sin la interfaz. No hay arranque automático de servicios.

## Primera prueba

Estos comandos crean directorios nuevos; cambia sus nombres para repetirlos.

```bash
# Entrena un LSTM pequeño durante 128 pasos del entorno.
.venv/bin/python -m ucl.cli train --config configs/EXP-002.json --seed 7 --steps 128 --output runs/mi-lstm

# Evalúa el modelo guardado y graba el episodio.
.venv/bin/python -m ucl.cli evaluate --config configs/EXP-002.json --checkpoint runs/mi-lstm/checkpoints/last.pt --task development-navigation --seed 7 --output runs/mi-lstm-evaluacion

# Continúa desde pesos, optimizador, estado recurrente y RNG guardados.
.venv/bin/python -m ucl.cli resume --config configs/EXP-002.json --checkpoint runs/mi-lstm/checkpoints/last.pt --steps 128 --output runs/mi-lstm-continuacion
```

`128 pasos` significa 128 interacciones con el entorno, **no 128 épocas**. A2C reúne un pequeño tramo de experiencia y actualiza los pesos. La configuración inicial usa tramos de 16 pasos. En `resume`, `--steps` añade experiencia al estado restaurado.

Para continuar exactamente un entrenamiento usa `last.pt`: guarda los últimos pesos. Durante el entrenamiento, `best.pt` se elige exclusivamente por validación. Si no hay mejora, puede contener los mismos pesos que `initial.pt`. En una ejecución de sólo evaluación no hay nueva selección por aprendizaje. Tener `initial.pt`, `best.pt` y `last.pt` **no garantiza que el agente haya aprendido una habilidad**. La arquitectura seleccionada debe ser compatible con el checkpoint. La versión actual guarda también RNG de NumPy y la identidad del entorno; `resume` rechaza un entorno distinto. El crecimiento permitido se restaura explícitamente.

## Cambiar y comparar arquitecturas

| Archivo | Arquitectura |
|---|---|
| `EXP-001.json` | Agente aleatorio, sin aprendizaje |
| `EXP-002.json` | Encoder pequeño + LSTM |
| `EXP-003.json` | Encoder pequeño + GRU |
| `EXP-004.json` | Encoder pequeño + RIM inspirado en el artículo |
| `EXP-005.json` | RIM expandible, crecimiento programado |
| `EXP-006.json` | LLM local congelado, baseline de acciones |
| `EXP-007.json` | LLM congelado + RIM entrenable |
| `EXP-008.json` | LLM congelado + RIM expandible |
| `CUSTOM-EXAMPLE.json` | Componente propio registrado como plugin |

Los controles `CONTROL-GROWTH-NONE`, `CONTROL-GROWTH-PLATEAU` y `CONTROL-RIM-MAX` separan crecimiento desactivado, crecimiento por meseta y capacidad máxima desde el inicio. `OPTIONAL-LARGER-LLM` está marcado como no disponible; no se descargó ese modelo.

```bash
.venv/bin/python -m ucl.cli train --config configs/EXP-003.json --seed 7 --steps 128 --output runs/mi-gru
.venv/bin/python -m ucl.cli compare runs/mi-lstm runs/mi-gru --output runs/comparacion-lstm-gru.json
```

El informe compara resultados y muestra si coinciden tareas, semillas de evaluación y experiencia. No mezcles un entrenamiento continuado de 256 pasos con uno de 128 como si recibieran la misma experiencia.

Para ejecutar la matriz neuronal breve con dos semillas, de forma secuencial:

```bash
.venv/bin/python -m ucl.cli matrix --seeds 7,19
```

`--include-llm` añade EXP-006/007/008 y consume más recursos. Consulta primero su estado real en el informe. Los resultados del LLM no se deducen de los resultados neuronales sin LLM. EXP-006 usa una observación numérica y no dispone de memoria ni información privilegiada. Es una referencia básica; no mide la capacidad máxima del LLM.

## Aprendizaje entre tareas

```bash
# Aprende A, luego B/C/D, y vuelve a medir A.
.venv/bin/python -m ucl.cli train --config configs/EXP-002.json --curriculum return_to_previous --seed 7 --output runs/retencion-lstm

# Secuencia de mecánicas, composiciones y tareas reservadas.
.venv/bin/python -m ucl.cli train --config configs/EXP-003.json --curriculum progressive --seed 7 --output runs/progresivo-gru

# Compara experiencia previa frente a una red con la misma inicialización.
.venv/bin/python -m ucl.cli learning-to-learn --config configs/EXP-002.json --seed 7 --output runs/aprender-a-aprender-lstm
```

Los currículos fijan sus pasos por etapa en `configs/tasks.json`; `--steps` no reemplaza esos valores. Las tareas de desarrollo entrenan. Validación selecciona checkpoints. Las tareas reservadas sólo se evalúan. Las evaluaciones no cambian pesos ni el estado del entrenador.

Si el agente obtiene cero antes y después, **no demuestra que conserve memoria**: puede no haber aprendido la tarea. Lee las definiciones y límites en [SCIENCE.md](SCIENCE.md).

## Crear tu componente

Ya existe un ejemplo ejecutable:

```bash
.venv/bin/python -m ucl.cli train --config configs/CUSTOM-EXAMPLE.json --seed 7 --steps 32 --output runs/mi-componente
```

1. Usa `src/ucl/custom_example.py` como referencia. La clase hereda de `Component` y se registra con `register_component(nombre, clase)`.
2. Declara dimensiones y estado recurrente. `forward` devuelve `(representación, estado_siguiente, señales)`.
3. En una configuración nueva añade `"plugins": ["ucl.mi_componente"]` y cambia el `type` del nodo por su nombre registrado.
4. Conserva las interfaces; no necesitas modificar el runner, XLand, evaluación, MLflow ni visualización.

Las ramas se declaran con `inputs` y una unión explícita `merge: concat` o `merge: sum`. Si las dimensiones no coinciden, añade un adaptador como nodo. No se descartan estados ni se reajustan tensores silenciosamente.

## Dónde se guarda todo

Cada ejecución crea una carpeta en `runs/` con:

- Configuración, versiones, semillas, tareas y snapshot del código.
- `metrics.jsonl`, `progress.json`, `summary.json` y muestras de recursos.
- Checkpoints `initial.pt`, `best.pt` y `last.pt` cuando hay modelo; un fallo puede añadir `interrupted.pt`.
- Episodios con acciones, recompensas, señales reales e imágenes `frames.npz` verificadas por SHA-256.

**Replay** reproduce esas imágenes sin ejecutar modelos. **Resume** carga el entrenamiento para continuarlo. Son operaciones distintas.

[MLflow — Universal Cognitive Lab](http://127.0.0.1:5000/#/experiments/12/runs?workflowType=machine_learning) usa un namespace separado en el servicio existente. Si el servicio no está disponible, el cliente guarda en `runs/mlruns-isolated/`. No inicia ni cambia la configuración del servidor.

## Recursos y alcance

- Instalación independiente en este directorio y su propia `.venv`, `pyproject.toml` y `uv.lock`.
- XLand ejecuta JAX en CPU. Las redes pequeñas usan CPU por defecto; los ejemplos con Qwen usan CUDA.
- Equipo observado: RTX 4070 Laptop con 8188 MiB VRAM y aproximadamente 24 GB RAM en Windows (CIM: 24,784,028 kB). WSL dispone de aproximadamente 6 GB RAM y 8 GB swap. La solicitud indicaba 16 GB; se documenta la medición real sin cambiar límites del sistema.
- Una ejecución del laboratorio a la vez. `resources` permite fijar tiempo, RAM libre mínima, temperatura y espacio. Los controles se comprueban entre operaciones; no son interrupciones de una llamada en curso.
- No hay APIs de pago, nube ni descarga automática de modelos. Qwen se referencia desde el modelo local ya instalado.
- El RIM de este laboratorio es una implementación pequeña **inspirada en RIM**, con desviaciones documentadas. No es una nueva réplica exacta del artículo ni reutiliza los checkpoints RIM antiguos. Véase [RIM.md](RIM.md).
- La conservación del laboratorio anterior se verifica contra el inventario inicial, incluyendo datos históricos de MLflow; el informe distingue lo comprobado de lo pendiente.

Para una investigación más larga, copia una configuración dentro de este laboratorio, define la hipótesis, controles, presupuesto y criterio de éxito, y ajusta sus pasos y límites. Aumentar el tiempo por sí solo no convierte una prueba técnica en un descubrimiento.

## Verificación

```bash
.venv/bin/python -B -m pytest --basetemp audit/pytest-local
python3 -B audit/preserve_existing.py verify
```

Las pruebas de UI usan grabaciones artificiales identificadas como fixtures; los resultados científicos proceden únicamente de `runs/`. El inventario inicial y la verificación final están en `audit/`.

Documentación: [entorno](ENVIRONMENT.md), [RIM](RIM.md), [ciencia](SCIENCE.md), [interfaz](DASHBOARD.md), [informe](IMPLEMENTATION_REPORT.md).
