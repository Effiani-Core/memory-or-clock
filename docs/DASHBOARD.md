# Interfaz visual

Abre http://127.0.0.1:8501/. La interfaz organiza el trabajo en cuatro vistas. Si hay ejecuciones terminadas de algún estudio, abre en Estudios; si no, en Simulación.

## Estudios: cada estudio en visual

Es la parte visual de MLflow: MLflow guarda los números; aquí ves la curva y lo que hace el agente en el mapa. La vista encuentra sola cualquier estudio que cumpla el formato de abajo; no hace falta tocar la interfaz para un estudio nuevo.

1. **Ejecución:** estudio (si hay más de uno), fase, tarea entrenada, arquitectura y semilla. Esa combinación es una ejecución de MLflow; si hay varias, aparece un selector con la fecha de cada una (abre en la más reciente).
2. **Resultado:** curva de éxito en validación con el criterio de adquisición, la línea del agente al azar y el punto en que la tarea se dio por adquirida. Al lado, la tabla antes y después de entrenar, con los paneles que midió el estudio. «Abrir en MLflow» lleva a la misma ejecución.
3. **Ver al agente en el entorno:** eliges checkpoint, tarea y mapas, y «Grabar episodios» los graba en segundo plano. Si existe la otra arquitectura con la misma tarea y semilla, se graba también para verlas lado a lado.
4. **Episodios grabados:** cada episodio se abre en Simulación, con su pareja de comparación si existe.

Reglas de la grabación:

- Los pesos no cambian y la ejecución original no se modifica. Cada grabación queda en `runs/ui-*-study-*` con un `recording.json`. Una grabación nunca se guarda dentro de otra ejecución, de otra grabación ni de una copia de MLflow.
- Cada mapa usa la misma regla de muestreo que las evaluaciones de paneles del estudio, así que el episodio coincide con el que contó en su tabla. En la fase de un mapa de STUDY-001, la curva repitió el mapa 1 con otro muestreo, así que su episodio puede no coincidir con el de la curva.
- Los paneles sellados (por defecto **final**) quedan cerrados hasta la evaluación final del estudio y no se pueden grabar.
- Se comprueba el hash del checkpoint y que los archivos del estudio sean los mismos con los que se entrenó. Si cambiaron, la ejecución se puede ver pero no grabar. Si algo falla, el trabajo queda como fallido con «Reintentar»; no hay reintentos automáticos.
- «En la mano» es un dato del observador (lo que el agente sostiene) y no forma parte de la entrada del modelo.

Enlace directo a una ejecución: `http://127.0.0.1:8501/?view=estudio&run=<carpeta dentro de runs/>`, por ejemplo `run=study_001/20261017T041659887950Z-pilot-S001-LSTM-s7-B`. Los enlaces viejos con solo el nombre de la carpeta siguen funcionando si ese nombre es único; si hay varias con ese nombre, o la ejecución no existe o no terminó, la vista lo dice en vez de abrir otra.

La misma grabación se puede lanzar desde WSL:

```bash
.venv/bin/python -m ucl.cli study-record --run study_001/<carpeta> --checkpoint after_B --task B --panel test --maps 1,2,3 --output runs/<nombre-nuevo>
```

### Formato de un estudio

Un estudio aparece solo cuando tiene estas dos piezas:

| Pieza | Dónde | Qué debe tener |
| --- | --- | --- |
| **Definición** | `configs/<nombre>/` | `protocol.json` con `study_id` y `tasks` (clave de tarea → origen), y opcionalmente `sealed_panels` (por defecto `["final"]`). `tasks.json` con el mismo contrato que `configs/tasks.json`. `panels.json` con `task_ids` (clave → id de tarea) y `panels` (id de tarea → panel → lista de semillas de mapa). Las claves de tarea y los nombres de panel son nombres simples (letras, números, `_ . -`). |
| **Ejecución** | cualquier carpeta dentro de `runs/` | `summary.json` con `study_id`, `protocol_hash`, `status`, `architecture_id`, `architecture`, `phase`, `task_keys`, `init_seed` y `checkpoints` (nombre → `path` y `sha256`). Para la curva y la tabla: `curves`, `acquisition`, `matrix` y `random_baselines`, como los escribe STUDY-001. |

El `protocol_hash` es el sha256 de los bytes de `protocol.json`, `tasks.json` y `panels.json`, en ese orden. Así cada ejecución se une a la definición exacta con la que se entrenó. Las copias de MLflow (`mlartifacts`, `mlruns-isolated`) y los registros de trabajos no cuentan como ejecuciones.

## Simulación: replay grabado

- **Panel izquierdo:** selecciona arquitectura, task y episodio guardado; el run de origen está en el contexto del episodio.
- **Mapa central:** muestra el episodio. El objetivo aparece junto al mapa; la flecha roja representa al agente.
- **Inspector del paso:** muestra la acción, recompensa y señales registradas en ese instante.
- **Controles de reproducción:** permiten avanzar una acción, reproducir o volver al inicio. La barra de progreso indica el paso actual; «Saltar al paso» es una selección manual independiente.

Es un **replay grabado**: verifica checksums y muestra imágenes guardadas, sin ejecutar inferencia ni entrenamiento. Si comparas agentes, se sincronizan por paso del entorno; un episodio terminado conserva su última imagen.

## Configurar evaluación

La configuración se divide en tres tarjetas:

1. **Modelo:** arquitectura y checkpoint compatible. El checkpoint indica qué pesos se usarán y su ejecución de origen.
2. **Escenario:** entorno, task o currículo, split y seed. El objetivo describe qué debe resolver el agente.
3. **Ejecución:** revisa la configuración y lanza la evaluación. Los pesos permanecen fijos; al terminar quedan resultados y episodios grabados.

Si no eliges un checkpoint, la interfaz indica que se inicializarán los componentes entrenables. En arquitecturas con un modelo de lenguaje congelado, éste conserva sus pesos originales. Evaluar no entrena esos componentes.

## Resultados

Mantén separados los dos tipos de datos:

- **Evaluación:** tasa de éxito, recompensa media y número de episodios. Corresponden al run y a la task seleccionados.
- **Entrenamiento:** experiencia acumulada, curvas y pérdidas disponibles. Describe el proceso que actualizó los pesos; sus resultados no sustituyen la evaluación.

La arquitectura, el checkpoint y la ejecución permiten identificar de dónde sale cada resultado. Un paso de entrenamiento y un paso dentro del episodio mostrado tienen contextos distintos: no deben interpretarse como el mismo contador.

## Qué significa cada término

| Término | Significado |
| --- | --- |
| **Arquitectura** | Estructura del agente: componentes y conexiones. Por ejemplo, un encoder conectado a una LSTM. |
| **Checkpoint** | Estado guardado del modelo, incluidos sus pesos, que permite cargar una versión concreta. |
| **Entorno** | Mundo y dinámica con los que interactúa el agente; aquí, XLand. |
| **Task** | Tarea concreta: objetivo y reglas que debe resolver. |
| **Split** | Uso reservado de los datos: development para desarrollar, validation para seleccionar y held-out para evaluar al final. |
| **Seed** | Semilla que determina la inicialización aleatoria. No identifica un mapa por sí sola: también importa la configuración del entorno y la tarea. |
| **Run / ejecución** | Una ejecución registrada de entrenamiento o evaluación, con configuración, métricas y archivos. |
| **Episodio** | Una interacción completa del agente con la tarea, desde el reinicio hasta terminar o alcanzar el límite. Una ejecución puede contener varios episodios. |
| **Paso** | Una acción del agente y la respuesta del entorno. Un episodio contiene varios pasos. |

La relación es: **arquitectura + checkpoint → agente evaluado; entorno + task + split + seed → escenario; ejecución → episodios y métricas**.

**Entrenar** actualiza pesos usando experiencia. **Evaluar** mide con pesos fijos. **Reproducir** muestra una grabación. El entrenamiento se lanza con los comandos del README; esta interfaz lanza evaluaciones y grabaciones de estudios.

## Compatibilidad e interpretación

El catálogo ofrece checkpoints compatibles con la arquitectura y configuración elegidas; la carga valida el archivo. **last** es el último estado guardado de la ejecución; **best** es el seleccionado por validación y puede coincidir con el inicial; **initial** es el estado al comenzar esa ejecución y puede contener aprendizaje anterior. Ninguno garantiza una habilidad aprendida.

La comparación visual exige igual tarea, hash de configuración y semilla, y distinta arquitectura. Esto no basta para una comparación científica justa si cambian el entrenamiento, los recursos o la configuración. Una prueba breve que termina correctamente tampoco demuestra una mejora científica.

El mapa completo, el objetivo y las reglas descritos son información para el observador. No se añaden a la entrada del modelo. Las acciones y señales proceden de la grabación; no se inventan pensamientos ni confianza.

Los registros con procedencia provisional declarada se omiten del catálogo normal y permanecen en disco. No se borran modelos, registros ni métricas.

## Operación local

Cada evaluación o grabación queda en `runs/ui-*`, con proceso y registros en `runs/ui_jobs/`. Sólo se permite una ejecución de la interfaz a la vez; cerrar la pestaña no cancela la evaluación.

Para iniciar la interfaz dentro de WSL, desde la carpeta del laboratorio:

```bash
.venv/bin/python -m ucl.cli dashboard
```

El servidor es local y no debe exponerse a Internet. No modifica el servicio MLflow del laboratorio anterior.
