# Ciencia del laboratorio

La hipotesis es que combinar mecanismos neuronales puede mejorar cuanto aprende
un agente, cuanto conserva y con cuanta experiencia lo consigue. Puede ser falsa.
Ninguna arquitectura es obligatoria ni ganadora por definicion.

Una prueba breve verifica que las piezas funcionan juntas. No demuestra AGI,
novedad, superioridad de RIM ni aprendizaje de las tareas. Los resultados validos
son los de ejecuciones guardadas.

## Volver a una tarea anterior

Aprender A; aprender B, C y D; volver a medir A sin entrenarla otra vez.
A es navegacion; B recoge objetos; C transforma al recoger; D combina al colocar.
Son reglas reales de XLand.

Sea S_A el exito en A despues de aprender A, y S_D el exito en esos mismos
mapas reservados despues de aprender B/C/D:

| Medida | Calculo | Lectura |
|---|---|---|
| Retencion | Exito final S_D | Cuanto resuelve al volver |
| Olvido desde A | max(0, S_A - S_D) | Cuanto perdio |
| Transferencia hacia atras | S_D - S_A | Positivo: mejora; negativo: deterioro |
| Caida acumulada tras B/C/D | max(0, S_A - S_D) | Es la misma caida final que el olvido; no identifica cuando ni por que ocurrio |
| A en mapas nuevos | Exito con otras semillas | Dependencia del mapa concreto |

Las tasas se guardan en 0..1; multiplicar por 100 da puntos porcentuales. El olvido
desde el mejor resultado intermedio es otra medida y debe etiquetarse aparte.
La razon final/inicial es indefinida cuando el resultado inicial es cero.
Si nunca aprendio A, cero antes y cero despues no demuestran buena memoria.

Las evaluaciones no actualizan pesos. Los mapas reservados son iguales entre
arquitecturas y distintos de los de entrenamiento. El estado recurrente se
reinicia por episodio; los pesos conservan el aprendizaje entre tareas.
La transferencia hacia atras sigue la idea de medir el cambio en tareas anteriores
tras aprender otras, usada en [GEM](https://arxiv.org/abs/1706.08840).
Aqui medimos episodios resueltos, no exactitud de clasificacion.

## Aprender a aprender

Comparar dos condiciones de la misma arquitectura:

1. Con experiencia: aprende otras familias y despues la familia nueva.
2. Sin experiencia: restaura los pesos iniciales exactos y aprende esa familia.

Ambas reciben igual experiencia en la nueva tarea, las mismas condiciones y los
mismos mapas de evaluacion. Medimos exito frente a interacciones, incluyendo el
punto anterior al primer entrenamiento. Guardamos la curva, su area normalizada
y las interacciones necesarias para superar un umbral fijado previamente.

El criterio implementado es al menos 75% de exito en dos evaluaciones consecutivas.
El momento de cumplimiento es el segundo punto, sin retrocederlo al primero. Si no llega,
se registra no alcanzado dentro del presupuesto, nunca un tiempo inventado.
Cuatro mapas sirven para verificar el calculo, no para una conclusion robusta.
El coste de la experiencia previa se reporta ademas del coste en la nueva tarea.
Se cuentan las interacciones reales de entrenamiento previo, entrenamiento objetivo
y todas las evaluaciones (curvas, validacion y evaluacion final). Las mismas semillas
de evaluacion se verifican en cada punto de ambas curvas. Igual semilla inicial de
entrenamiento no implica visitar los mismos mapas durante el mismo tiempo: las
acciones y la duracion de los episodios pueden diferir entre condiciones.

El optimizador tambien conserva experiencia. Para atribuir un efecto solo a los
pesos hace falta un control que reinicie el optimizador. Si la red crecio antes,
tambien hay que controlar esa capacidad adicional.
Una mejora en una familia es transferencia en esa prueba. No demuestra una
capacidad general de metaaprendizaje. Hacen falta varias familias, semillas,
dificultades comparables y curvas que muestren aprendizaje real.

## Comparacion de crecimiento

| Configuracion | Control |
|---|---|
| EXP-005 | Crecimiento en un momento fijado |
| CONTROL-GROWTH-NONE | Red expandible con crecimiento desactivado |
| CONTROL-GROWTH-PLATEAU | Crecimiento por una meseta medida |
| CONTROL-RIM-MAX | Red fija con la capacidad maxima desde el comienzo |

La capacidad maxima separa el efecto de tener mas parametros del de decidir
cuando agregarlos. Comparar parametros reales: igual numero de modulos no
garantiza igualdad. Igualar tareas, semillas, interacciones, optimizador, encoder,
cabezas y evaluacion. Reportar memoria, latencia, coste y eventos de crecimiento.
Una meseta no prueba falta de capacidad; puede reflejar mala exploracion.
La politica inicial no se presenta como novedad cientifica.

## Las medidas solicitadas

| Medida | Cobertura y limite |
|---|---|
| In-Domain Generalization | Nuevos mapas de la misma tarea; configurado |
| OOD | Colores/rulesets reservados y una regla direccional nueva; cambios concretos |
| Cross-Domain Generalization | Pendiente un segundo dominio cientifico |
| Compositional Generalization | Reglas compuestas funcionan; falta reservar combinaciones de primitivas aprendidas |
| Einstellung effect | Pendiente una tarea donde una estrategia aprendida obstaculice otra mejor |
| Negative Transfer | Experiencia previa empeora la curva frente al control sin experiencia |
| Positive Transfer | Experiencia previa mejora esa curva con los mismos controles |
| Retencion | Volver a A tras B/C/D sin entrenar A otra vez |
| Plasticidad | Adquirir tareas posteriores; cambiar pesos no basta |

El curriculo progressive entrena tambien una composicion y reserva variaciones
de colores. No aisla combinaciones nunca entrenadas. Fallar o transferir mal
tampoco basta para demostrar el efecto Einstellung.

## Cuando interpretar resultados

- Desarrollo para implementar; validacion para seleccionar; held-out para informar
  el resultado final. No ajustar ni entrenar con held-out.
- Varias inicializaciones para rendimiento: resultados individuales, media,
  variacion y diferencias emparejadas. Las corridas existentes usan 7 y 19.
  Para investigaciones futuras se proponen al menos tres semillas; no se presentan
  semillas propuestas como resultados ya obtenidos.
- Si todos quedan en cero, comprobar en desarrollo que la tarea permite aprender.
  Cero olvido no significa retener algo que nunca aprendio.
- Si todos llegan al maximo, la prueba no distingue bien las hipotesis.
- Comparar latencia con igual renderizado/registro; separar tiempo del modelo y
  tiempo total. Registrar fracasos y limites de recursos.
- No entrenar horas para llenar una tabla. Primero verificar una prueba capaz de
  distinguir lo que queremos contrastar.

Fuentes: [RIM](https://arxiv.org/abs/1909.10893),
[XLand-MiniGrid](https://arxiv.org/abs/2312.12044),
[codigo oficial](https://github.com/dunnolab/xland-minigrid).
La integracion sola no demuestra novedad.
