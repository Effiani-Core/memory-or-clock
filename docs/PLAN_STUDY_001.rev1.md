# STUDY-001 — Aprendizaje y retención

**Fecha:** 16 de octubre de 2026. **Estado:** plan; no se ha iniciado este estudio.

Este documento concreta la propuesta del usuario sobre el laboratorio existente. Los números de entrenamiento y evaluación son decisiones propuestas para el estudio, no resultados obtenidos ni valores exigidos por un artículo.

## 1. Objetivo y alcance

**Pregunta:** ¿qué habilidades adquieren LSTM, GRU y nuestro RIM durante experiencias sucesivas, cuánto conservan y qué coste tiene hacerlo?

**Secuencia:** aprender A → aprender B → aprender C → evaluar A/B/C sin volver a entrenarlas.

El primer resultado será una referencia reproducible, aunque ninguna arquitectura gane. No exige inventar ahora otro componente. Permitirá decidir después qué mecanismo propio vale la pena investigar.

Aquí se mide conservación del aprendizaje en los pesos. El estado recurrente se reinicia por episodio. Recordar datos durante esperas largas o recuperar experiencias episódicas necesitaría otro protocolo. El RIM de este laboratorio es una implementación inspirada en el artículo; este estudio no es una réplica del paper.

Usaremos `study_id=UCL-STUDY-001`. No se renombra `EXP-001`: ese identificador ya corresponde a la arquitectura aleatoria. Tampoco se alteran los bloques históricos 01/02/03 del laboratorio anterior.

## 2. Punto de partida comprobado

| Elemento | Estado observado |
|---|---|
| Entorno | XLand-MiniGrid integrado; reset, acciones, recompensas, render y semillas |
| Redes | Agente aleatorio, LSTM, GRU, RIM fijo y expandible disponibles |
| Aprendizaje | A2C funcional; actualiza pesos y conserva el optimizador entre tareas |
| Evidencia | MLflow, métricas locales, episodios grabados y checkpoints |
| Resultados actuales | Pruebas breves de infraestructura; no evidencia suficiente de dominio de tareas |
| Retención registrada | A: 0/4 después de A y 0/4 al regresar; retención no interpretable |

Evidencia local: `README.md`, `docs/IMPLEMENTATION_REPORT.md`, `docs/SCIENCE.md`, `runs/retention-final/summary.json` y configuraciones `EXP-*`.

Las configuraciones actuales usan normalmente 128 interacciones, dos episodios de evaluación y límites de prueba técnica. No se ejecutará la matriz actual como si fuera este estudio.

## 3. Tareas y separación de datos

| Tarea | Mecánica existente | Habilidad medida |
|---|---|---|
| A | `development-navigation` | Alcanzar la casilla objetivo |
| B | `development-pickup` | Recoger y conservar el objeto objetivo |
| C | `development-hold_transform` / `AgentHoldRule` | Recoger un objeto que se transforma en el objetivo |

C representa una regla de transformación concreta. No se presentará como capacidad general de descubrir reglas desconocidas.

Crear un manifiesto propio del estudio, conservando los originales. Para la medición principal mantener iguales reglas, colores, tamaño, observaciones y límite de pasos. Los `validation-*` y `heldout-*` actuales cambian también colores/reglas: se reservan para análisis OOD separado.

**Paneles propuestos por tarea:**

- 64 estados iniciales de entrenamiento.
- 32 de validación para las curvas y decisiones de desarrollo.
- 64 de prueba para medir antes/después sobre condiciones idénticas.
- 32 adicionales reservados para la evaluación final en mapas nuevos.
- Un subconjunto fijo de 16 mapas de entrenamiento se evalúa por separado como «mapas conocidos».

Verificar que existan suficientes estados distintos antes de entrenar. Semillas diferentes pueden producir el mismo mapa 5×5: deduplicar y separar por configuración física inicial, incluyendo objetos, posición y orientación del agente, sin incluir la semilla en esa identidad. Si no caben los paneles, reducirlos o ajustar el tamaño durante el piloto y documentarlo; nunca contar duplicados como mapas nuevos. Este dato privilegiado sólo sirve para organizar la evaluación, no entra en la red.

Evaluar el mismo panel reservado varias veces no entrena al agente. No usar sus resultados para ajustar hiperparámetros. Si se modifica el protocolo tras inspeccionar la prueba final, iniciar una nueva versión con paneles finales nuevos.

## 4. Preparación mínima antes de entrenamientos extensos

| Prioridad | Cambio | Comprobación necesaria |
|---|---|---|
| P0 | Guardar cada 1.000 interacciones o 120 segundos, lo que ocurra primero, al terminar una actualización | El checkpoint carga; conserva pesos, optimizador, RNG, topología, estado recurrente, observación y estado reconstruible del entorno |
| P0 | Guardar `after_A.pt`, `after_B.pt`, `after_C.pt` sin sobrescribirlos | Cada evaluación referencia archivo y hash exactos |
| P0 | Reanudar el currículo desde su etapa y pasos pendientes | Una interrupción breve reproduce la continuación de una ejecución CPU sin interrupción |
| P0 | Persistir currículo resuelto, hash, etapa, contador, evaluaciones completadas y criterio de selección | Reanudar no vuelve a empezar A ni consume dos veces el presupuesto |
| P0 | Medir A/B/C antes de entrenar y después de cada etapa | Tabla completa: checkpoint × tarea × semilla × panel |
| P0 | Separar checkpoint final y seleccionado por validación | Retención principal usa el estado real al terminar cada etapa; no recarga un best antiguo |
| P1 | Registrar curvas durante cada tarea, recursos y agrupación correcta de resultados | Las semillas de una red expandible permanecen juntas aunque terminen con tamaños distintos |

Archivos previstos: `src/ucl/runner.py`, `checkpoints.py`, `reports.py`, configuraciones nuevas del estudio y pruebas focalizadas. No se sustituye el motor de entrenamiento ni se reconstruye el dashboard.

**Problemas confirmados que esto resuelve:** hoy se guarda al terminar etapas; `resume` no conserva el cursor del currículo; la selección de best usa la tarea de validación configurada, normalmente navegación; la evaluación final usa el modelo actual. `--steps` no reemplaza los presupuestos internos de un currículo. El manifiesto actual de retención incluye D, que no forma parte de este plan A/B/C.

## 5. Orden de ejecución y límites

Las siguientes fases tienen criterios de salida. No son rondas artificiales de entrenamiento ni requieren detenerse cada cierto número de epochs.

| Fase | Trabajo | Resultado necesario para continuar |
|---|---|---|
| 1. Preparación | Implementar y probar las correcciones P0 | Guardado, reanudación y métricas fiables |
| 2. Curva inicial | Evaluar el azar y entrenar LSTM en A; comprobar B y C individualmente | Aprendizaje verificable en cada tarea |
| 3. Referencias | Ejecutar A→B→C con LSTM y GRU | Curvas, checkpoints y matriz de retención completos |
| 4. Modularidad | Repetir con RIM fijo bajo el mismo protocolo | Comparación con presupuesto y capacidad controlados |
| 5. Crecimiento | Ejecutar los controles de expansión | Separar efecto del crecimiento del de tener más parámetros |
| 6. Análisis | Revisar resultados por semilla, fallos, coste y retención | Informe que indique qué se observó y qué hipótesis sigue |

**Piloto propuesto:** hasta 20.000 interacciones por tarea e inicialización; evaluación cada 2.000 interacciones sobre los 32 mapas de validación. Empezar con LSTM, sin LLM y sin crecimiento. Usar las inicializaciones 7, 19 y 31 para verificar reproducibilidad. A/B/C individuales suman como máximo 180.000 interacciones de entrenamiento si se ejecutan todas las combinaciones; las evaluaciones se contabilizan aparte.

**Criterio operativo propuesto de adquisición:** al menos 75% de éxito en dos evaluaciones consecutivas y una mejora de al menos 20 puntos porcentuales frente al modelo inicial y al agente aleatorio. El cumplimiento se registra en la segunda evaluación, sin retrocederlo a la primera. Son criterios prácticos del piloto, no garantías estadísticas.

Si una tarea no resulta aprendible en ese presupuesto, realizar un diagnóstico y como máximo una revisión documentada de tarea/hiperparámetros en desarrollo. Si vuelve a fallar, detener la comparación completa y emitir el resultado del diagnóstico. No aumentar automáticamente horas ni ejecutar todas las arquitecturas sobre una tarea sin aprendizaje demostrado.

**Estudio exploratorio:** tres inicializaciones emparejadas; tope provisional de 50.000 interacciones por tarea. El presupuesto común definitivo se fija con el piloto antes de comparar arquitecturas; no se elige por cuál favorece al RIM. Todos completan el mismo presupuesto aunque alcancen antes el umbral. Registrar el instante de adquisición permite comparar eficiencia.

Con 50.000 interacciones por tarea, la comparación LSTM/GRU/RIM fijo supondría 9 ejecuciones y 1,35 millones de interacciones de entrenamiento. El bloque de crecimiento se contabiliza por separado: hasta 12 ejecuciones más si se requieren sus cuatro brazos. No es una estimación de horas ni una orden para lanzar esa cantidad ahora. Medir segundos por interacción y coste de evaluación en el piloto antes de fijar el calendario.

Una sola ejecución a la vez; XLand y las redes pequeñas inicialmente en CPU. Guardar y pausar ante límites de disco/RAM/temperatura o fallos, conservando el motivo. No hacen falta descargas, servicios de pago ni una LLM.

## 6. Comparaciones justas

**Comparación central:** aleatorio, LSTM, GRU y RIM fijo. El aleatorio se evalúa; no recibe entrenamiento.

Mantener encoder, entrada, acciones, algoritmo, protocolo, mapas y presupuesto de interacción. Ajustar las dimensiones de memoria para aproximar parámetros entrenables totales dentro de ±10% del presupuesto inicial elegido. Registrar los tamaños reales; si no se puede igualar, tratar la capacidad como un factor adicional y no atribuir toda diferencia a la modularidad. Mantener dimensiones de entrada/salida y usar adaptadores explícitos sólo si son necesarios.

Mismos números de semilla emparejan condiciones; no significan pesos idénticos entre arquitecturas distintas. Usar el mismo presupuesto de búsqueda de hiperparámetros. Si se usa una sola configuración A2C, las conclusiones serán sobre esa configuración, no sobre el máximo potencial de cada arquitectura.

**Bloque de crecimiento:**

1. RIM expandible con crecimiento desactivado.
2. Mismo RIM con una ampliación en un punto predefinido, inicialmente la transición A→B.
3. Mismo RIM con política de meseta, fijada usando sólo desarrollo.
4. RIM fijo con capacidad máxima desde el inicio.

Limitar inicialmente a 3→4 módulos, una ampliación como máximo, y mantener 2 módulos activos. Emparejar exactamente la inicialización de los tres primeros brazos. El control de capacidad máxima tiene más parámetros desde el inicio y debe reportarse así. Reutilizar un control ya ejecutado sólo si configuración, pesos iniciales, protocolo y presupuesto son realmente equivalentes.

El `EXP-005` actual crece por calendario; no aprende cuándo crecer. No se usará su intervalo de prueba de dos actualizaciones como decisión científica. Una meseta tampoco demuestra por sí sola falta de capacidad. Registrar tanto efectos sobre aprendizaje como módulos, parámetros y coste.

## 7. Métricas e interpretación

Sea S(j,t) la tasa de éxito en la tarea j al terminar la etapa t.

| Medida | Definición y uso |
|---|---|
| Resultado principal | Cambio en A: 100 × [S(A,C) − S(A,A)], en puntos porcentuales |
| Retención final | S(A,C), junto a S(A,A); una cifra final aislada no demuestra conservación |
| Olvido de A | 100 × max(0, S(A,A) − S(A,C)) |
| Plasticidad | Éxito adquirido en B y C; conservar A sin aprender lo nuevo no satisface el objetivo |
| Task Success Rate | Porcentaje de episodios resueltos en cada tarea, no porcentaje de tipos de tarea |
| Sample Efficiency | Interacciones hasta el umbral y área normalizada bajo la curva; no alcanzado si agota presupuesto |
| Transferencia hacia atrás | Cambio con signo en tareas ya aprendidas; mostrar A y B por separado |
| Crecimiento | Eventos, módulos, parámetros reales y momento en que se añaden |
| Coste | Interacciones de entrenamiento y evaluación por separado, tiempo, RAM, memoria de GPU si aplica y latencia |

La matriz antes/después permite explorar rendimiento en B/C antes de entrenarlas. **No basta para afirmar que aprenderá más rápido gracias a experiencia previa.** Esa afirmación exige los controles desde cero de STUDY-002.

Comparar siempre el éxito inicial en A, el final y su cambio: una caída menor puede deberse a haber aprendido menos A al principio. No declarar superioridad por el olvido aislado; comprobar también rendimiento final y adquisición de B/C. Comparar cada arquitectura con LSTM mediante diferencias emparejadas por inicialización. Mostrar todos los resultados, media y dispersión; no tratar 64 mapas de una misma red como 64 entrenamientos independientes. Tres semillas ofrecen una primera caracterización, no garantizan precisión suficiente. Un resultado prometedor requiere una confirmación predefinida con al menos cinco inicializaciones y un conjunto de prueba sin usar; no seleccionar sólo las mejores corridas.

Si A no se adquirió, la retención es **no interpretable**. Si B/C no se aprendieron, no hay evidencia de conservación acompañada de capacidad de aprender. Conservar esos fracasos en el informe; no eliminarlos para mejorar promedios. Si todos saturan el máximo, no hay evidencia de ventaja en esta dificultad. Un resultado nulo o negativo también completa el estudio.

## 8. Entregables y organización

- Protocolo y configuraciones congelados bajo `configs/study_001/`.
- Nuevas ejecuciones bajo `runs/study_001/`; registros antiguos intactos.
- MLflow en el namespace existente `Universal Cognitive Lab`, con etiquetas `study_id`, fase, arquitectura, semilla, hash del protocolo y checkpoint evaluado.
- Curvas de A/B/C; matriz de éxito antes/después; comparación de retención, adquisición y coste; episodios representativos seleccionados con una regla fija.
- Checkpoints iniciales, periódicos y de final de cada etapa, con prueba de recarga.
- Informe final con resultados por semilla, fracasos, limitaciones y próxima hipótesis.
- Comandos exactos de ejecución al implementar los nuevos archivos. No presentar como ejecutables comandos que dependen de configuraciones aún inexistentes.

El dashboard sirve para observar y examinar resultados. No demuestra por sí solo que una arquitectura haya aprendido. No se necesitan cambios grandes de interfaz para este estudio.

## 9. STUDY-002 queda separado

Después de obtener aprendizaje confiable, comparar la misma arquitectura con experiencia previa frente a la misma inicialización sin esa experiencia, aprendiendo una familia nueva con igual presupuesto de adaptación. Controlar además optimizador y capacidad; informar por separado el coste previo. El helper actual de learning-to-learn impone crecimiento desactivado y 1.200 segundos: habrá que hacer explícitas esas decisiones antes de reutilizarlo.

Una mejora en una familia sería evidencia de transferencia en ese caso. La capacidad general de aprender mejor requeriría varias familias y controles adicionales.

## 10. Referencias y límites

- [GEM — métricas de aprendizaje continuo y transferencia](https://arxiv.org/abs/1706.08840): referencia conceptual para separar desempeño y transferencia; aquí usamos éxito en episodios.
- [XLand-MiniGrid](https://arxiv.org/abs/2312.12044): entorno original. Nuestro currículo pequeño es un protocolo local, no el benchmark completo del artículo.
- [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893) y `docs/RIM.md`: origen del mecanismo y desviaciones de la implementación local.

Este plan establece una referencia experimental. No promete AGI, superioridad modular ni un resultado publicable por sí solo.