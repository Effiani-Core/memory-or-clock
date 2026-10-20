# Informe final — Universal Cognitive Lab

**Laboratorio implementado y probado. Las fases 0–11 están completadas. No se ha demostrado una mejora científica de las arquitecturas.**

Corte de evidencia: `2026-10-15T05:45:35.926615+00:00`. Código final: `ab503095fa1144496fd91f062e3e1fd98e0eb8f80e00e92d44fd5cbd83fa09b9`.

## Instalación y conservación del laboratorio anterior

Ubicación: `/home/researcher/ai-research-lab/labs/universal_cognitive_lab`, WSL `AIResearchLab`, usuario `researcher`. Tiene entorno `.venv`, `pyproject.toml`, `uv.lock`, código, configuraciones y resultados propios. No se reinstaló ni se refactorizó el laboratorio principal.

La [auditoría final](../audit/verification-20261015T053750/report.json) terminó con exit 0. Frente a la línea base de 159,318 archivos, 16,178 directorios y 21 enlaces:

- 159,316 archivos conservaron su contenido; no hubo archivos ausentes ni cambios de permisos o metadatos en los archivos protegidos.
- Los otros dos cambios físicos corresponden al log del servicio MLflow y a su SQLite, por las nuevas ejecuciones autorizadas. No se cambió la configuración del servicio.
- Las **12,759 filas históricas** de MLflow siguen intactas. No cambió su esquema. Todas las nuevas filas pertenecen al namespace **Universal Cognitive Lab**, experimento 12: 28 runs nuevos en el servidor al cierre de la auditoría.
- No aparecieron rutas nuevas fuera del sublaboratorio. El HEAD principal sigue en `45e29eb9f74c9adac23a83131d21888d38e371a8`; se conservaron los cambios previos del usuario.

Se verificaron 55,094,188,771 bytes lógicos. Los enlaces duros hacen que esa suma supere el espacio físico. `.git` se comprueba por HEAD, estado y diffs; no se presenta como una imagen bit a bit de un repositorio activo.

## Qué se entregó y comprobó

| Fase | Entrega | Evidencia |
|---|---|---|
| 0 — Inventario | Archivos, paquetes, hardware, servicios y filas MLflow previas | `audit/baseline.json` y verificación final |
| 1 — Aislamiento | Python 3.12.15 y 126 paquetes en entorno propio | `audit/dependency_inventory.json`, lockfile |
| 2 — Núcleo | Registro, contratos de dimensiones/estado, grafo con ramas y componentes propios | Suite integral y `custom-smoke` |
| 3 — XLand | Adaptador oficial, reset/step/recompensa/fin, reglas, seeds, render y observación pública | Pruebas y episodios reales |
| 4 — Redes pequeñas | MLP, LSTM y GRU; A2C recurrente con gradientes | EXP-002/003, cambio de pesos y checkpoints |
| 5 — Modularidad | RIM inspirado y crecimiento acotado; controles sin crecimiento, programado, por meseta y capacidad máxima | EXP-004/005 y controles, dos semillas |
| 6 — LLM congelado | Qwen local solo, con RIM y con RIM expandible | EXP-006/007/008 reales; hashes congelados iguales |
| 7 — Aprendizaje entre tareas | Retorno a A, currículo progresivo y comparación con/sin experiencia previa | `retention-final`, `progressive-final`, `learning-to-learn-smoke` |
| 8 — Interfaz | Streamlit, evaluación, replay, pausa, paso, velocidad, reinicio y comparación sincronizada | `audit/browser_validation.json` y captura |
| 9 — MLflow | Namespace propio, métricas, parámetros, checkpoints, episodios y comparación | `audit/mlflow_final_api.json` |
| 10 — Integración | Entrenar → guardar → cargar → evaluar → reproducir; componente propio sin editar runner | Suite integral, UI EXP-008 y resume real |
| 11 — Cierre | Pruebas, auditoría previa/final, resultados e instrucciones | Este informe e inventarios |

La matriz utiliza XLand sobre JAX CPU. La política sólo recibe la observación pública egocéntrica; reglas, objetivo oculto y mapa completo quedan fuera de sus entradas. Los frames completos son herramientas del evaluador. Hay 19 tareas declaradas: 6 de desarrollo, 6 de validación y 7 reservadas.

A2C reúne tramos de 16 interacciones, conserva estados recurrentes y actualiza pesos entrenables. La tasa inicial es 0.001 y gamma 0.99. Evaluar no actualiza pesos ni altera el estado de entrenamiento. El control de semillas mantiene un pool de entrenamiento separado de evaluación. Los currículos continúan con la misma red entre tareas.

RIM significa aquí **RIM-inspired**: entrada selectiva con token nulo, mecanismos GRU independientes, activación dispersa y comunicación. Las diferencias respecto al artículo están en [RIM.md](RIM.md). No se presenta como réplica exacta ni reutiliza los checkpoints RIM del laboratorio antiguo.

La expansión conserva pesos y estados compatibles de Adam, añade parámetros nuevos, registra topología y respeta límites. La meseta se mide con recompensa; también puede indicar mala exploración. No demuestra por sí sola que falte capacidad.

## Pruebas y restauración

**133 pruebas aprobadas, 0 fallos, 0 errores y 0 omitidas; exit 0.** Evidencia: [JUnit](../audit/test_suite_1.xml), [recibo](../audit/test_suite_1_receipt.json) y log. La suite tardó 84.431 s; el proceso completo, 99.09 s. Incluye entorno real, contratos, gradientes, crecimiento/optimizador, selección por validación, ausencia de información privilegiada, RNG, replay, tracking e interfaz.

No se cambió código ni configuración después de la suite final. Los archivos coinciden con el snapshot de `resume-smoke`, `retention-final` y `progressive-final`.

- **Resume real:** cargó `learning-to-learn-smoke/experienced/checkpoints/last.pt`; los pesos iniciales coincidieron con el final anterior. Continuó de 512 a 528 interacciones y cambió pesos. Se guardaron 16 pasos nuevos. Evidencia: `audit/resume_validation.json`.
- **Topología expandida:** los tests restauran módulos y optimizador y comparan la siguiente actualización en CPU. No se carga silenciosamente una forma incompatible.
- **Evaluación desde UI:** cargó EXP-008 con cuatro módulos desde `last.pt`. Coincidieron pesos entrenables, pesos congelados, tamaño expandido, observaciones y las 32 acciones del episodio original. Evidencia: `audit/frozen_restore_and_ui.json`.
- **Replay:** utiliza las imágenes guardadas y su checksum; no vuelve a ejecutar inferencia ni entrenamiento. La comparación avanza por paso del entorno.
- **Guardado:** los modelos tienen `initial.pt`, `best.pt` y `last.pt`; los hashes y tamaños figuran en `audit/documentation_results_snapshot.json`. El agente aleatorio no tiene pesos que guardar. `best.pt` depende sólo de validación y puede seguir siendo el inicial si no mejoró.

Los checkpoints guardan modelo, topología, optimizador, estado recurrente, entorno reconstruible y RNG de Python/NumPy/Torch/CUDA. Los checkpoints tempranos sin los campos añadidos durante el desarrollo no se presumen compatibles con resume estricto final; conservan su snapshot y pueden usarse para evaluación cuando su arquitectura es compatible. No se garantiza identidad numérica entre distintos dispositivos o versiones CUDA.

## Experimentos reales

Todas las filas siguientes tienen `summary.json` FINISHED. No se cuentan los fixtures de tests ni duplicados de MLflow. Éxito reservado es una proporción entre 0 y 1, con muy pocas muestras.

| Run | Arquitectura | Semilla | Interacciones de entrenamiento acumuladas | Éxito reservado | Crecimientos |
|---|---|---:|---:|---:|---:|
| [20261015T045902032296Z-EXP-002](../runs/20261015T045902032296Z-EXP-002/summary.json) | EXP-002 | 7 | 32 | 0.0 | 0 |
| [20261015T050306928204Z-EXP-001](../runs/20261015T050306928204Z-EXP-001/summary.json) | EXP-001 | 7 | 0 | 0.0 | 0 |
| [20261015T050407892348Z-EXP-002](../runs/20261015T050407892348Z-EXP-002/summary.json) | EXP-002 | 7 | 128 | 0.0 | 0 |
| [20261015T050551208430Z-EXP-003](../runs/20261015T050551208430Z-EXP-003/summary.json) | EXP-003 | 7 | 128 | 0.0 | 0 |
| [20261015T050651692156Z-EXP-004](../runs/20261015T050651692156Z-EXP-004/summary.json) | EXP-004 | 7 | 128 | 0.0 | 0 |
| [20261015T050712024523Z-EXP-005](../runs/20261015T050712024523Z-EXP-005/summary.json) | EXP-005 | 7 | 128 | 0.0 | 1 |
| [20261015T050731812946Z-CONTROL-RIM-MAX](../runs/20261015T050731812946Z-CONTROL-RIM-MAX/summary.json) | CONTROL-RIM-MAX | 7 | 128 | 0.0 | 0 |
| [20261015T050751357444Z-CONTROL-GROWTH-NONE](../runs/20261015T050751357444Z-CONTROL-GROWTH-NONE/summary.json) | CONTROL-GROWTH-NONE | 7 | 128 | 0.0 | 0 |
| [20261015T050808845782Z-CONTROL-GROWTH-PLATEAU](../runs/20261015T050808845782Z-CONTROL-GROWTH-PLATEAU/summary.json) | CONTROL-GROWTH-PLATEAU | 7 | 128 | 0.0 | 1 |
| [20261015T050827976737Z-EXP-001](../runs/20261015T050827976737Z-EXP-001/summary.json) | EXP-001 | 19 | 0 | 0.0 | 0 |
| [20261015T050849257635Z-EXP-002](../runs/20261015T050849257635Z-EXP-002/summary.json) | EXP-002 | 19 | 128 | 0.0 | 0 |
| [20261015T050914677425Z-EXP-003](../runs/20261015T050914677425Z-EXP-003/summary.json) | EXP-003 | 19 | 128 | 0.0 | 0 |
| [20261015T050930942340Z-EXP-004](../runs/20261015T050930942340Z-EXP-004/summary.json) | EXP-004 | 19 | 128 | 0.0 | 0 |
| [20261015T051006208170Z-EXP-005](../runs/20261015T051006208170Z-EXP-005/summary.json) | EXP-005 | 19 | 128 | 0.0 | 1 |
| [20261015T051029078888Z-CONTROL-RIM-MAX](../runs/20261015T051029078888Z-CONTROL-RIM-MAX/summary.json) | CONTROL-RIM-MAX | 19 | 128 | 0.0 | 0 |
| [20261015T051102101755Z-CONTROL-GROWTH-NONE](../runs/20261015T051102101755Z-CONTROL-GROWTH-NONE/summary.json) | CONTROL-GROWTH-NONE | 19 | 128 | 0.0 | 0 |
| [20261015T051127554335Z-CONTROL-GROWTH-PLATEAU](../runs/20261015T051127554335Z-CONTROL-GROWTH-PLATEAU/summary.json) | CONTROL-GROWTH-PLATEAU | 19 | 128 | 0.0 | 1 |
| [custom-smoke](../runs/custom-smoke/summary.json) | CUSTOM-EXAMPLE | 7 | 32 | 0.0 | 0 |
| [frozen-llm-006-smoke](../runs/frozen-llm-006-smoke/summary.json) | EXP-006 | 7 | 0 | 0.0 | 0 |
| [frozen-llm-007-smoke](../runs/frozen-llm-007-smoke/summary.json) | EXP-007 | 7 | 8 | 0.0 | 0 |
| [frozen-llm-008-smoke](../runs/frozen-llm-008-smoke/summary.json) | EXP-008 | 7 | 8 | 0.0 | 1 |
| [progressive-final](../runs/progressive-final/summary.json) | EXP-003 | 7 | 768 | 0.0 | 0 |
| [resume-smoke](../runs/resume-smoke/summary.json) | EXP-002 | 7 | 528 | 0.0 | 0 |
| [retention-final](../runs/retention-final/summary.json) | EXP-002 | 7 | 512 | 0.0 | 0 |
| [retention-smoke](../runs/retention-smoke/summary.json) | EXP-002 | 7 | 512 | 0.0 | 0 |
| [ui-20261015T053018Z-56398959-00](../runs/ui-20261015T053018Z-56398959-00/summary.json) | EXP-008 | 240000 | 0 | 0.0 | 0 |
| [learning-to-learn-smoke/experienced](../runs/learning-to-learn-smoke/experienced/summary.json) | EXP-002-experienced | 7 | 512 | 0.0 | 0 |
| [learning-to-learn-smoke/fresh_same_initialization](../runs/learning-to-learn-smoke/fresh_same_initialization/summary.json) | EXP-002-fresh_same_initialization | 7 | 256 | 0.0 | 0 |

La matriz neuronal contiene **16 ejecuciones**, dos semillas (7 y 19), con 128 interacciones por red y cero entrenamiento para el agente aleatorio. Todas obtuvieron 0/2 en la evaluación reservada. [Comparación](../runs/neural-matrix-comparison.json).

LSTM tiene 11,367 parámetros; GRU, 9,255; RIM fijo, 31,191. El RIM expandible pasó de 3 a 4 módulos y llegó a 40,103 parámetros; el control fijo máximo tiene 40,103 desde el inicio. Esto permite comparar crecimiento con capacidad final equivalente, pero no convierte todas las familias en modelos de igual tamaño. Los informes separan cohortes incompatibles y muestran experiencia, semillas y procedencia.

**Cambiar pesos prueba que se ejecutó aprendizaje por gradientes. No prueba que se haya aprendido una habilidad.** No hay ventaja científica demostrada en esta matriz breve.

### Retención y progresión

`retention-final` entrenó LSTM 128 pasos en A, B, C y D: **512 en total**. A obtuvo 0/4 inmediatamente después y 0/4 al regresar. La razón de retención es **indefinida** porque no se demostró capacidad inicial. Cero olvido no significa memoria perfecta. B obtuvo 1/2 y A en otros cuatro mapas, 2/4; son resultados locales, sin conclusión estadística.

`progressive-final` entrenó GRU **768 pasos** en seis familias. Recogida y regla cambiada obtuvieron 1/2; composición y mecánica direccional reservadas, 0/2. Las evaluaciones posteriores mantuvieron el contador en 768. La secuencia funciona; no se demuestra transferencia composicional ni aprendizaje de una mecánica desconocida.

### Experiencia previa frente a red fresca

[Informe L2L](../runs/learning-to-learn-smoke/learning_to_learn.json). Ambos agentes parten de la misma inicialización. El experimentado recibe 256 pasos previos y luego 256 en la nueva tarea; el control recibe sólo los 256 de la nueva tarea. Comparten tarea, pool y semilla inicial de entrenamiento y mapas de evaluación. Sus acciones y duración de episodios producen **distinta exposición efectiva a mapas**; se registra la semilla de cada interacción, sin afirmar secuencias idénticas.

| Medición | Fresco | Con experiencia |
|---|---:|---:|
| Entrenamiento previo | 0 | 256 |
| Entrenamiento en tarea objetivo | 256 | 256 |
| Interacciones de evaluación | 755 | 888 |
| Total entorno, entrenamiento + evaluación | 1,011 | 1,400 |
| Área normalizada bajo curva de éxito | 0.28125 | 0.25 |

Diferencia: **−0.03125**, sin ventaja de la experiencia previa en esta ejecución. Ninguno alcanzó 75% en dos evaluaciones consecutivas. Una semilla no demuestra metaaprendizaje ni transferencia negativa general. La experiencia incluye pesos y estado del optimizador; su coste previo se cuenta.

### LLM local

Se utilizó el Qwen2.5-1.5B-Instruct ya instalado, revisión `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16 CUDA. No se descargó otro modelo. Sus **1,543,714,304 parámetros permanecieron congelados**, comprobados por hash completo antes y después de EXP-006/007/008.

EXP-006 realizó cero pasos de entrenamiento. EXP-007 y EXP-008 realizaron 8 cada uno: cambiaron sus componentes entrenables y mantuvieron el LLM intacto. EXP-008 creció de 3 a 4 módulos, de 78,743 a 87,655 parámetros entrenables. Los tres obtuvieron 0/1 reservado. Los pesos grandes se referencian por manifiesto verificado, sin duplicarlos en cada checkpoint.

EXP-006 es una referencia básica de selección de acciones: observación numérica, sin memoria de episodios ni información privilegiada. No mide la capacidad máxima del LLM ni demuestra que sea incapaz en general.

## Interfaz y MLflow

[Abrir interfaz](http://127.0.0.1:8501/) · [Abrir métricas](http://127.0.0.1:5000/#/experiments/12/runs?workflowType=machine_learning).

La interfaz quedó ejecutándose durante la entrega. Se probó selección, evaluación de checkpoint real, comparación de LSTM/GRU con igual tarea/semilla, reproducción, pausa, paso, velocidad y reinicio. Se inspeccionó ancho 1200 px y panel estrecho; se restauró el tamaño normal. [Captura del resultado](../audit/dashboard-evaluation.jpg).

La API de MLflow confirmó para `progressive-final` estado FINISHED, **89 métricas y 12 parámetros**, además de directorios de checkpoints y episodios, configuración, fuentes y resúmenes. Dos ejecuciones tempranas (EXP-001/002, semilla 7) usaron `runs/mlruns-isolated/` al fallar el acceso al servidor; no se presentan como visibles en el servidor. La interfaz y los informes locales sí leen sus resultados.

## Dependencias y archivos

Código: `src/ucl/` (19 módulos); configuraciones: `configs/`; pruebas: `tests/`; documentos: `docs/`; resultados y checkpoints: `runs/`; pruebas de conservación y recibos: `audit/`.

La lista de archivos entregados con tamaño y SHA-256 está en `audit/deliverable_files.json`; el inventario completo de los 126 paquetes está en `audit/dependency_inventory.json`. Versiones principales: Python 3.12.15, Torch 2.11.0+cu128, NumPy 2.5.4, XMiniGrid 0.9.3, JAX/JAXlib 0.4.38 CPU, Flax 0.10.2, Chex 0.1.88, Transformers 4.57.6, Accelerate 1.12.0, MLflow-skinny 3.17.0, Streamlit 1.66.0, pytest 9.1.1.

El entorno reutilizó descargas en caché mediante copia, sin enlaces compartidos con la `.venv` principal. Ocupa aproximadamente 11 GB por Torch/CUDA; las nuevas descargas de dependencias fueron aproximadamente 225 MB. No se usaron APIs de pago ni servicios de cómputo en nube.

## Límites y funcionalidad no completada

| Estado | Alcance |
|---|---|
| Implementado y probado | Todos los criterios de infraestructura, modularidad, entrenamiento, congelación, crecimiento, currículos, checkpoints, replay, interfaz y MLflow de las fases 0–11 |
| Implementado, sin eficacia científica demostrada | Crecimiento por meseta, memoria útil, transferencia positiva/negativa, mejora por experiencia y generalización; las pruebas breves no bastan |
| No disponible por recursos/modelo instalado | Baseline LLM mayor: configuración explícita sin resultados; un 7B BF16 representativo excede 8 GB VRAM |
| Futuro, no implementado como adaptador | Otros simuladores, robótica, otros dominios; la interfaz permite añadirlos pero sólo XLand se integró |
| Fuera de esta validación | AGI, novedad/publicabilidad, Einstellung, transferencia entre dominios y combinaciones nunca entrenadas con controles científicos completos |
| Archivo auxiliar omitido | Un JSON redundante de protocolo fue rechazado por revisión automática, sin razón detallada. No se intentó eludirlo; el protocolo está documentado en `SCIENCE.md` y no bloquea funcionalidad |

Hardware observado: RTX 4070 Laptop, 8188 MiB VRAM; Windows reportó aproximadamente 24 GB RAM (24,784,028 kB), aunque la solicitud decía 16 GB. WSL dispone de aproximadamente 6 GB RAM y 8 GB swap. No se alteraron controladores ni límites globales. Una ejecución a la vez; las redes pequeñas usan CPU y Qwen CUDA. Los límites de recursos se comprueban entre operaciones y no interrumpen una llamada ya en curso. Hubo timeouts transitorios de comunicación WSL bajo presión; se recuperó sin reiniciar el laboratorio ni detener servicios ajenos.

## Incidencias resueltas y procedencia

Un script de documentación sobrescribió por error **sólo** el `summary.json` de la evaluación nueva de UI. Se restauraron sus bytes exactos desde el artefacto de MLflow; coincidían también con `progress.json`. Se verificaron pesos, topología y episodio. Evidencia: `audit/metadata_repair.json`. Ningún checkpoint ni archivo anterior fue alterado por ese error.

`progressive-smoke` se ejecutó mientras cambiaba código antes de congelar la versión final: su snapshot no corresponde por completo al código cargado. Se conserva como diagnóstico provisional y se excluye de la tabla final. Se repitió con código estable en `progressive-final`; también se cerró `retention-final`. Evidencia: `audit/execution_provenance.json`. Las ejecuciones anteriores conservan sus versiones; no se presentan como ejecutadas con la última revisión.

## Comandos de uso

Desde PowerShell, entra a WSL:

```powershell
wsl -d AIResearchLab -u researcher
```

Dentro de WSL:

```bash
cd /home/researcher/ai-research-lab/labs/universal_cognitive_lab
# Si la interfaz ya está abierta, usa el enlace; para iniciarla de nuevo:
.venv/bin/python -m ucl.cli dashboard
```

Para entrenar, evaluar y comparar, usa carpetas nuevas:

```bash
.venv/bin/python -m ucl.cli train --config configs/EXP-002.json --seed 7 --steps 128 --output runs/mi-lstm
.venv/bin/python -m ucl.cli evaluate --config configs/EXP-002.json --checkpoint runs/mi-lstm/checkpoints/last.pt --task development-navigation --seed 7 --output runs/mi-lstm-evaluacion
.venv/bin/python -m ucl.cli train --config configs/EXP-003.json --seed 7 --steps 128 --output runs/mi-gru
.venv/bin/python -m ucl.cli compare runs/mi-lstm runs/mi-gru --output runs/comparacion-lstm-gru.json
.venv/bin/python -m ucl.cli resume --config configs/EXP-002.json --checkpoint runs/mi-lstm/checkpoints/last.pt --steps 128 --output runs/mi-lstm-continuacion
```

Para crear una arquitectura propia, toma `src/ucl/custom_example.py`, hereda de `Component`, declara dimensiones/estado y registra tu clase con `register_component`. Añade el módulo a `plugins` de una configuración y usa su nombre en `type`. El ejemplo probado se ejecuta así:

```bash
.venv/bin/python -m ucl.cli train --config configs/CUSTOM-EXAMPLE.json --steps 32 --output runs/mi-componente
```

[README](../README.md) contiene matriz, currículos y aprendizaje a aprender. Para un ensayo posterior más largo, copia una configuración dentro de este laboratorio y declara primero hipótesis, control, número de semillas, presupuesto y criterio de descarte. Ajusta `train_steps`/`--steps` y límites de recursos; los currículos toman sus pasos de `configs/tasks.json`. Esta entrega no deja entrenamientos largos ejecutándose.

## Revisión de interfaz — 15 de octubre de 2026

Después de la entrega inicial se simplificó la pantalla por solicitud del usuario. Ahora hay tres vistas: Ver un agente, Probar un modelo y Resultados. Se explican modelos, objetivos, acciones y aciertos; los detalles técnicos quedan plegados. Sólo se ofrecen versiones guardadas compatibles. El código de entrenamiento y los resultados previos no se cambiaron.

Esta revisión aprobó 24 pruebas de interfaz/catálogos y Ruff. Una evaluación real del LSTM completó el recorrido desde la pantalla hasta su grabación, sin modificar pesos. Evidencia: `audit/ui-simplification-final-receipt.json`. La suite integral de 133 pruebas del informe anterior corresponde a la entrega inicial; esta revisión volvió a ejecutar las pruebas de las áreas cambiadas.


## Revisión de diseño — 15 de octubre de 2026

Se conserva la terminología técnica y se cambia la jerarquía visual: selección del episodio, mapa e inspector; tarjetas de modelo/escenario/ejecución; evaluación separada del entrenamiento. Arquitectura, checkpoint, task, split, seed, run y episodio tienen etiquetas y definiciones. Tema claro local consistente.

El visor distingue replay de evaluación nueva. La comparación exige identidad de tarea (hash), semilla e ID de tarea. Se corrigió un conflicto entre los timers de Streamlit y el slider: posición canónica independiente, controles fuera del fragmento animado y selección manual sin escritura automática.

Validación de esta revisión: 34 pruebas focalizadas aprobadas y Ruff correcto (`audit/ui-layout-verified.xml`, `audit/ui-layout-verified.log`). Revisión en navegador de configuración, resultados y reproducción. No se ejecutó entrenamiento ni una evaluación nueva; sólo se reprodujeron episodios existentes. Se reinició únicamente el servidor del dashboard para validar la interfaz.
