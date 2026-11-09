# Comparación de alertas por señal y detector sobre d1_c1_b_up

Fecha de ejecución: 2026-11-02 (America/Bogota).

## Pregunta y diseño

El experimento compara las alertas que producen Page-Hinkley, ADWIN y KSWIN sobre
el conteo de comentarios y las medidas ER, EMD y MEC. La referencia es
[`d1_c1_b_up.json`](../configs/experiments/d1_c1_b_up.json). La comparación cubre doce
pares señal–detector en tres corridas: cada corrida aplica un mismo detector a
las cuatro señales, con una instancia independiente por señal. Cada corrida
realiza su propia asociación temporal, proyección de candidatos y construcción
local de evidencia.

Las alertas de cada par son el resultado principal. Las asociaciones y los
candidatos describen las coincidencias dentro de cada corrida; no representan
las 81 asignaciones posibles de detectores entre las cuatro señales.

## Resultados

Las tres corridas finalizaron y procesaron 78.016 comentarios cada una. Los
conteos de alertas fueron:

| Señal | Page-Hinkley | ADWIN | KSWIN |
|---|---:|---:|---:|
| Conteo de comentarios | 151 | 290 | 272 |
| ER | 1 | 6 | 52 |
| EMD | 1 | 6 | 53 |
| MEC | 0 | 2 | 52 |

Page-Hinkley produjo 153 candidatos, ADWIN 302 y KSWIN 412. La construcción
local de evidencia terminó en las tres corridas y ningún candidato quedó
pendiente de horizonte temporal. Se conservaron 886 alertas y 867 candidatos.

Las observaciones de actividad, polarización, distribuciones y posturas
coinciden byte por byte con los artefactos de `d1_c1_b_up`. Las cuatro rutas de
la referencia reproducen sus 53.314 decisiones, valores y metadatos al comparar
sin las identidades de ejecución. Cada alerta aparece exactamente una vez en
una asociación y un candidato, con su correspondiente paquete de evidencia.

Las cuatro rutas de ADWIN y las cuatro de KSWIN también reproducen todas las
decisiones de sus respectivas corridas anteriores. La corrección de referencia
solo cambia los resultados de Page-Hinkley. El consumo agregado fue de cero
tókens, con cero ejecuciones de G-1/G-2, cero llamadas externas y cero fugas
futuras registradas. El bloqueo de red no registró intentos de conexión.

## Controles de la referencia

La referencia corregida usa Page-Hinkley con `mode="up"`, tal como el cuadro
compartido por el usuario: 151 alertas de actividad, 6 de ER con ADWIN,
1 de EMD con Page-Hinkley y 52 de MEC con KSWIN. La comparación anterior con
`mode="both"` se conserva en [su propio registro](d1_c1_b_detector_comparison.md).

Se conservan los datos, el tema, la proposición, las posturas, el período, el reloj,
las ventanas, las cadencias, las políticas, los parámetros de los tres detectores
y las opciones de evidencia. Solo cambian las rutas señal–detector, el `run_id`
y las rutas de salida. La configuración resuelta de la referencia con el código
actual coincide con el `resolved_config.json` de la corrida original.

| Control | Valor conservado |
|---|---|
| Fuente | `data/silver/experiments/d1/d1.csv`, política `causal_minimal_v1` |
| Tema y proposición | `c1_ivan_cepeda`, `C1` |
| Posturas | `outputs/stances/d1_c1_stance/comment_stances.jsonl` |
| Período UTC | 2026-02-08T17:05:53Z a 2026-04-21T18:09:15Z |
| Paso del replay | 1 minuto |
| Conteo | Ventana de 15 minutos; cadencia de 3 minutos |
| ER, EMD y MEC | Ventana de 90 minutos; cadencia de 15 minutos |
| Distribución | `likert5_author_latest_classified_v1` |
| Page-Hinkley | `mode="up"`; los demás parámetros se conservan |
| Ausencia de señal | `preserve_state_v1` |
| Asociación | `bounded_temporal_clusters_v1`, tolerancia de 5 minutos |
| Evidencia | `immediate_causal`, lookback de 10 minutos, lookahead de 0 segundos |
| Estado de parámetros | `DEVELOPMENT_NOT_CALIBRATED` |

SHA-256 verificados antes de ejecutar:

- Dataset: `b17710bdb7222740f539f6dea13f19f85b87d7259cad6d832e560a86057bb10c`.
- Posturas: `e69d8f565f9d6bb17dab9100e2aa166b87dbb90cd4bd00573639492a3b87471c`.

El ejecutor valida además la proposición, el contexto, los identificadores,
los autores y los tiempos. Libera cada postura junto con su comentario original.
Las posturas se reutilizan; no se ejecuta clasificación.

## Ejecución sin consumo de tókens

Las tres corridas usan `--execute --skip-rag`. La opción conserva la construcción
local de evidencia y pasa `execute_rag=False` al ejecutor existente. G-1, G-2 y sus
proveedores quedan sin ejecutar. La opción `rag.g2_gate.mode=OFF` se conserva por
identidad con la referencia; la desactivación de RAG depende de `--skip-rag`.

La ejecución realizada añadió un bloqueo de conexiones y resolución de nombres
mediante un audit hook de CPython. Los manifiestos permiten comprobar por separado
la ausencia de validaciones, consumo de tókens, llamadas externas y fugas futuras.

## Reproducción

Ejecutar desde la raíz del repositorio, con el entorno y las dependencias del
proyecto. Las carpetas de salida deben estar ausentes o vacías; el ejecutor evita
sobrescribir resultados existentes.

```bash
.venv/bin/python scripts/run_causal_end_to_end.py \
  --config configs/experiments/d1_c1_b_up_page_hinkley.json \
  --base-dir . --execute --skip-rag

.venv/bin/python scripts/run_causal_end_to_end.py \
  --config configs/experiments/d1_c1_b_up_adwin.json \
  --base-dir . --execute --skip-rag

.venv/bin/python scripts/run_causal_end_to_end.py \
  --config configs/experiments/d1_c1_b_up_kswin.json \
  --base-dir . --execute --skip-rag
```

Cada corrida queda en `outputs/causal_end_to_end/<run_id>/`. Allí se conservan
las observaciones, `detection_results.jsonl`, `detection_alerts.jsonl`,
`detection_associations.jsonl`, `event_candidates.jsonl`, `candidate_evidence/`,
`resolved_config.json` y `causal_replay_manifest.json`.

El resumen y la auditoría de la comparación quedan en
`outputs/causal_end_to_end/d1_c1_b_up_detector_comparison/`. Los resultados generados
y los datasets permanecen fuera de Git conforme a la política del repositorio.

El [reporte local](../outputs/causal_end_to_end/d1_c1_b_up_detector_comparison/comparison_report.md)
enlaza los doce archivos de alertas y detalla la cobertura, las primeras y últimas
alertas, las asociaciones y los estados de evidencia. El archivo
`comparison_summary.json` conserva las métricas y verificaciones en formato
estructurado. `execution_audit.json` registra las tres corridas y sus tiempos;
`environment.json` y `requirements.freeze.txt` identifican el entorno usado.

## Límites de interpretación

Una alerta indica que el detector activó su criterio sobre esa serie numérica.
Su número no mide aciertos ni demuestra eventos reales. Las escalas del conteo,
ER, EMD y MEC son distintas, y los parámetros se conservan sin calibrarlos.
En particular, cero alertas es un resultado válido con esta parametrización.
El experimento no asigna precisión, exhaustividad ni superioridad a un detector.
