# Semántica de las señales de actividad de referencia

**Estado:** CURRENT CONTRACT
**Autoridad:** métricas y señales de actividad, `ActivitySignalDefinition`,
`ActivityObservation`, enrutamiento señal→detector, `DetectionResult`,
`EventCandidate`, `quality` y `detector_metadata`.
**Contexto arquitectónico:** [arquitectura end-to-end](pipeline_architecture.md).

Este documento fija el comportamiento observable que sirve como referencia durante
la modularización de A6. No declara que las señales, detectores o parámetros sean
óptimos, universales o definitivos.

La cadena conceptual protegida es:

```text
SOURCE
→ NORMALIZATION
→ ACTIVITY METRIC
→ ACTIVITY SIGNAL
→ REFERENCE DETECTOR
→ EVENT CANDIDATE
```

Un trigger de estas rutas representa un evento candidato, no la confirmación de un
evento real externo.

## Identificadores semánticos

| `signal_id` conceptual | Rol actual |
|---|---|
| `comment_count_event_window_120s_step_30s` | Señal embebida que alimenta XIAO |
| `new_comment_count_local_day_daily` | Señal que alimenta el baseline diario |
| `comment_count_event_window_20m_per_comment` | Señal diagnóstica del replay preparado |
| `unique_author_count_event_window_20m_per_comment` | Señal diagnóstica del replay preparado |
| `unique_video_count_event_window_20m_per_comment` | Señal diagnóstica del replay preparado |

Estos nombres son documentales. No cambian columnas ni artefactos existentes.

`unique_author_count_event_window_120s_step_30s` continúa siendo una señal candidata:
no se implementa ni se incorpora a las rutas productivas en A6-1.

## Referencia XIAO

### Definición de la señal

| Propiedad | Semántica actual |
|---|---|
| Métrica | Conteo de comentarios retenidos en el buffer temporal |
| Fuente | Comentarios preparados entregados secuencialmente al detector |
| Scope | Flujo seleccionado completo; no existe agrupación interna por video |
| Unidad | Comentarios por observación/tick |
| Ventana de referencia | 120 segundos |
| Cadencia | 30 segundos |
| Base temporal | `event_time_utc`, interpretado en UTC |
| Orden requerido | Timestamps no decrecientes; el replay es responsable de ordenar |
| Intervalo en un tick `t` | Se retienen timestamps `>= t - 120s`; con input ordenado no hay timestamps futuros, por lo que el intervalo nominal es `[t-120s, t]` |

### Ticks y entradas

- El primer tick es el primer múltiplo global de 30 segundos que se encuentra en o
  después del primer comentario válido.
- Antes de insertar un comentario posterior, el detector emite todos los ticks
  pendientes estrictamente anteriores a su timestamp.
- Un comentario que llega exactamente en un tick se inserta antes de evaluar ese
  tick y queda incluido.
- Si transcurren ticks sin comentarios, estos se evalúan cuando un comentario
  posterior hace avanzar el reloj del detector.
- `finalize()` no genera ticks vacíos posteriores al último comentario.
- Un comentario expira solamente cuando su timestamp es estrictamente menor que
  `tick - 120s`. El borde izquierdo se conserva.

### Timestamps iguales en un borde de slide

La API del detector recibe registros uno a uno. El primer comentario procesado en
un timestamp que coincide con un tick provoca la evaluación de ese tick. Otros
comentarios procesados después con el mismo timestamp ya no se incorporan a esa
observación y aparecen en el tick posterior.

Esto se protege como comportamiento actual para detectar regresiones. No se adopta
como requisito metodológico definitivo. Cuando A6 separe explícitamente la señal del
detector deberá conservarlo para compatibilidad o proponer el cambio funcional de
forma explícita.

### XIAO como referencia

XIAO es `REFERENCE_DETECTOR` y `REGRESSION_ANCHOR`. Puede permanecer, coexistir con
otros detectores, deshabilitarse o reemplazarse mediante configuración en la
arquitectura futura. Su preservación durante A6 no lo convierte en detector final.

## Referencia diaria

### Definición de la señal

| Propiedad | Semántica actual |
|---|---|
| Métrica | Conteo de `comment_id` únicos asignados por primera vez al ciclo |
| Fuente | Comentarios preparados con `event_time_utc` válido |
| Scope | Corpus completo del ciclo de simulación |
| Unidad | Comentarios únicos por día local |
| Ventana | Día calendario local |
| Cadencia | Diaria |
| Timezone operacional | `America/Bogota` |
| Base temporal | Hora de publicación `event_time_utc`, no hora de ingesta |
| Intervalo | `[inicio del día local, inicio del día siguiente)` convertido a UTC |

Para un día de Bogotá sin cambio de offset, por ejemplo el 1 de junio de 2026, el
intervalo es `[2026-06-01T05:00:00Z, 2026-06-02T05:00:00Z)`.

Consecuencias:

- un comentario en el inicio UTC convertido queda incluido;
- un comentario en el extremo final queda asignado al día siguiente;
- duplicados posteriores del mismo `comment_id` no incrementan la señal;
- los comentarios de la ventana analítica siempre cumplen
  `event_time_utc < data_cutoff_utc`;
- un día configurado sin comentarios produce valor cero;
- “new” significa publicado en el día simulado, no recién observado o ingerido.

La ruta de referencia es:

```text
new_comment_count_local_day_daily
→ daily_frequency_baseline
→ event candidate
```

El baseline diario es también una referencia experimental, no un detector definitivo.

## Perfiles XIAO que no deben confundirse

| Rol | Procedencia | Señal | Detector | `v_min` | Interpretación |
|---|---|---|---|---:|---|
| `REFERENCE` | `run_20260602T180842Z` | `comment_count_event_window_120s_step_30s` | `xiao_ema` | 15 | Referencia histórica experimental que produjo 18 eventos |
| `COMPATIBILITY_DEFAULT` | Configuración vigente | `comment_count_event_window_120s_step_30s` | `xiao_ema` | 46 | Default actual que preserva comportamiento del código |

Ningún perfil implica optimalidad, universalidad ni transferibilidad a otra señal o
dataset. En particular, los parámetros XIAO no se transfieren automáticamente a una
futura señal de autores únicos.

## Calidad disponible para la señal candidata de autores

En la revisión local del Gold vigente realizada el 23 de agosto de 2026, los 57.725
comentarios tenían `author_id` no nulo y no vacío. Esto respalda la viabilidad de la
señal candidata en ese corpus, pero no elimina la necesidad de reportar cobertura de
autor y calidad por ejecución futura.

## Estado actual de esta formalización

Las fases posteriores a A6-1 ya incorporaron los contratos neutrales, extrajeron la
construcción de la señal de comentarios, añadieron la señal experimental de autores
únicos, declararon la ruta señal→detector y definieron `DetectionResult` y
`EventCandidate`.

Triggered results can now be projected into the same neutral `DetectionAlert`
contract used by polarization. The alert adds explicit `signal_family=ACTIVITY`,
run/config lineage, support, causal cutoff and a deterministic reference to the
persisted result. It does not replace the legacy `EventCandidate` path and does not
perform association. Current activity signals emit numeric observations rather than
explicit absence states, so `preserve_state_v1` does not change their historical
behavior. See [detection alerts and gap policy](detection_alerts_and_gap_policy.md).

Estos avances no significan que toda la ruta productiva use ya la frontera neutral:

- XIAO produce `DetectionResult`, pero su camino histórico continúa publicando
  `completed_triggers`;
- el enrutamiento configurable existe y está probado, pero el runner cíclico de
  compatibilidad todavía invoca la ruta histórica;
- `EventCandidate` existe como contrato interno y proyección compatible, pero aún no
  es la autoridad general que alimenta evidencia/RAG;
- el baseline diario conserva su evento especializado;
- no existe todavía otro detector integrado al pipeline.

No se modificaron parámetros ni resultados de referencia. Cero eventos continúa
siendo un resultado válido de una ejecución configurada.

## Contrato neutral incorporado en A6-2

`ActivitySignalDefinition` identifica la semántica estable de una señal mediante:

```text
signal_id
metric
source
scope
unit
window
cadence
time_basis
timezone
interval_policy
```

No contiene estrategia ni parámetros de detector. En consecuencia, la misma
definición puede entregarse posteriormente a XIAO, Page-Hinkley u otro adaptador sin
modificar su significado.

`ActivityObservation` representa un valor causal de esa definición mediante:

```text
signal
observation_time_utc
window_start_utc
window_end_utc
value
support_count
quality
```

Los tiempos son UTC conscientes de zona y el fin de la ventana no puede superar el
tiempo de observación. El contrato no incorpora configuración de ejecución, paths,
identidad de corrida, serialización ni parámetros del detector. Esos datos deben
seguir perteneciendo al orquestador, la configuración resuelta y la trazabilidad de
la ejecución.

## Separación incorporada en A6-3

`EventWindowCommentCountSignal` conserva el buffer, la cadencia y la construcción de
`comment_count_event_window_120s_step_30s`. Emite `ActivityObservation` en el mismo
orden temporal documentado en A6-1.

XIAO conserva `on_event()` como adaptador de compatibilidad, pero la actualización de
sus EMA y su decisión reciben ahora una observación explícita mediante
`on_observation()`. La ruta es:

```text
comentario
→ EventWindowCommentCountSignal
→ ActivityObservation
→ XiaoEMATriggerDetector.on_observation
→ trigger de referencia
```

La ventana y la cadencia de la referencia XIAO todavía se originan en
`XiaoEMAConfig` para preservar la configuración vigente. La ruta configurable
incorporada después no duplica esos valores: declara únicamente `signal_id` y
`detector_id`. El runner de compatibilidad todavía no usa esa ruta como despacho
general.

## Señal experimental de autores únicos incorporada en A6-4

La segunda métrica demuestra que la infraestructura temporal no está limitada al
volumen de comentarios:

```text
author_id
→ autores conocidos distintos en la ventana
→ unique_author_count_event_window_120s_step_30s
→ ActivityObservation
```

| Propiedad | Valor |
|---|---|
| Rol | `EXPERIMENTAL_SIGNAL` |
| Métrica | `unique_authors` |
| Fuente | `prepared_comments` |
| Scope | `selected_comment_stream` |
| Unidad | `authors/window` |
| Ventana | 120 segundos |
| Cadencia | 30 segundos |
| Base temporal | `event_time_utc` |
| Intervalo | Cerrado, igual a la señal de comentarios |

La señal reutiliza `EventWindowActivitySignal`, que mantiene el buffer, los ticks y
los límites temporales. `EventWindowCommentCountSignal` permanece como constructor
compatible sobre esa misma infraestructura.

### Política de `author_id`

- un ID conocido y no vacío participa en `nunique`;
- repeticiones del mismo ID cuentan una sola vez;
- `None`, valores nulos y cadenas vacías no se convierten en autores sintéticos;
- los comentarios sin autor permanecen en `support_count`;
- si la ventana contiene al menos un autor ausente, `quality` es
  `degraded_missing_author_id`;
- en otro caso, `quality` es `passed`.

El Gold local revisado el 23 de agosto de 2026 mantiene 57.725 de 57.725 valores de
`author_id` presentes. Esta cobertura no se asume para futuras ejecuciones.

A6-4 no conectó esta señal al flujo productivo. A6-5 demostró después, mediante el
contrato de ruta y pruebas, que una observación de autores puede llegar a XIAO sin
exponer filas fuente. Esa prueba arquitectónica no afirma que los parámetros del
detector sean válidos para autores únicos ni selecciona la combinación para un
perfil experimental.

## Composición explícita señal → detector en A6-5

La asociación de una señal con un detector se declara mediante
`ActivityDetectionRouteConfig`. La ruta contiene únicamente dos referencias:

```json
{
  "detection": {
    "activity_route": {
      "signal_id": "comment_count_event_window_120s_step_30s",
      "detector_id": "xiao_ema"
    },
    "xiao_ema": {}
  }
}
```

`signal_id` identifica la semántica completa de la observación y `detector_id`
selecciona una implementación registrada. La ruta no copia ventanas, umbrales ni
otros parámetros: `ActivitySignalDefinition` conserva la autoridad semántica de la
señal y `XiaoEMAConfig` conserva la autoridad metodológica de XIAO.

El despacho valida que la observación corresponda al `signal_id` declarado y entrega
exclusivamente `ActivityObservation` al detector. Por tanto, el detector no necesita
conocer filas de comentarios, `author_id`, datasets ni paths. La ruta diaria
`new_comment_count_local_day_daily → daily_frequency_baseline` permanece como
consumidor especializado para no alterar su contrato consolidado.

La declaración alternativa
`unique_author_count_event_window_120s_step_30s → xiao_ema` demuestra composición
arquitectónica. No valida que los parámetros históricos de XIAO sean adecuados para
esa señal ni constituye una evaluación experimental.

## Resultado neutral de detección en A6-6

Cada evaluación de una observación puede expresarse mediante `DetectionResult`:

```text
detector_id
signal_id
observation_time_utc
triggered
quality
score                 # opcional; no implica probabilidad
detector_metadata     # evidencia matemática específica
```

El contrato responde qué detector evaluó qué señal, cuándo lo hizo y si su criterio
se satisfizo. `score` es opcional porque no todos los métodos producen una magnitud
comparable ni una probabilidad. La semántica de ese valor debe documentarla cada
detector. `confidence` no forma parte del contrato común para evitar atribuir una
interpretación probabilística inexistente.

En XIAO, `score` corresponde actualmente a la razón entre EMA rápida y EMA lenta. El
volumen, ambas EMA, la razón, warmup y cooldown permanecen en `detector_metadata`:
son evidencia de XIAO, no propiedades universales de un detector. La apertura y el
cierre del trigger histórico, sus comentarios asociados y sus artefactos no cambian.

El baseline diario puede proyectar sus filas de puntuación al mismo contrato:

```text
detector_id             = daily_frequency_baseline
signal_id               = definición semántica de la señal configurada
observation_time_utc    = tiempo de la fila diaria
triggered               = trigger_candidate
quality                 = estado de calidad de la fila
score                   = ratio_to_baseline, cuando esté definido
detector_metadata       = baseline_mean, delta, cambio porcentual,
                          threshold y condiciones
```

Esta proyección queda diferida: A6-6 no modifica los artefactos diarios ni obliga al
baseline a migrar. Sus eventos existentes continúan siendo la salida compatible.

`DetectionResult` tampoco es un evento candidato. El resultado solo afirma que un
criterio estadístico se satisfizo para una observación. A6-6C incorporó una promoción
interna que añade identidad, intervalo causal y referencias mínimas de linaje. Los
comentarios y videos permanecen fuera del candidato y se incorporan en evidencia.
La promoción todavía no reemplaza los eventos retrospectivos o diarios activos.
Esta frontera evita que el detector conozca almacenamiento, datasets o validación.

### Autoridad de calidad y evidencia

| Campo | Autoridad | Significado |
|---|---|---|
| `value` | Productor de señal | Valor cuantitativo de actividad |
| `support_count` | Productor de señal | Cantidad de registros que respaldan la observación |
| `quality` | Productor de señal | Calidad o limitaciones de la observación construida |
| `triggered` | Detector | Cumplimiento del criterio estadístico del método |
| `score` | Detector | Magnitud opcional y específica del método |
| `detector_metadata` | Detector | Evidencia y estado estadístico específico |

`ActivityObservation.quality` es la única autoridad de calidad de la observación.
`DetectionResult.quality` la propaga sin reinterpretarla. El detector no conoce por
qué faltó un autor ni modifica `degraded_missing_author_id`; únicamente evalúa el
valor recibido.

Warmup, cooldown, ventanas adaptativas, historia insuficiente del método y estados
como EMA pertenecen a `detector_metadata`. Por ejemplo, una observación puede tener
`quality="passed"` mientras XIAO informa `warmup_complete=false`. Ese estado no
convierte la calidad en `warmup`.

Asimismo, `quality` no es confianza ni probabilidad de evento. `score` tampoco mide
calidad: su interpretación depende de `detector_id` y no se asume comparable entre
XIAO y detectores futuros. `detector_metadata` es inmutable dentro del resultado y
no debe contener configuración duplicada, paths, secretos, `RunConfig` ni estado
global del pipeline.

Esta separación también admite el baseline diario: la calidad de su señal se
propagaría, mientras baseline, delta, ratio, umbrales y condiciones pertenecerían a
la evidencia específica del detector. Su adaptación continúa diferida y sus
artefactos actuales no cambian.

## Promoción mínima a candidato en A6-6C

`EventCandidate` representa únicamente la promoción de un resultado disparado. Se
construye por composición:

```text
ActivityObservation
+ DetectionResult(triggered=true)
+ EventCandidateLineage
+ intervalo causal de evidencia
→ EventCandidate
```

El candidato conserva la observación y el resultado originales; por eso no vuelve a
declarar `quality`, `score`, `detector_metadata`, valor de señal ni parámetros. La
calidad visible en el candidato es una propiedad delegada al `DetectionResult`, cuya
igualdad con la observación se valida durante la promoción.

El linaje mínimo contiene `run_id`, `config_hash`, `dataset_ref` y una referencia
opcional al manifest. No copia `RunConfig`, el dataset ni configuraciones completas.
La definición de señal permanece disponible a través de la observación.

La promoción recibe `candidate_id` explícitamente. No introduce una fórmula nueva:
un adaptador puede suministrar el `event_id` o `daily_event_id` histórico y preservar
su identidad vigente. El lifecycle es opcional (`point`, `open` o `closed`) porque el
baseline diario es puntual mientras XIAO conserva apertura/cooldown/cierre.

Los comentarios, videos, context units, consultas y resultados de validación no
forman parte del candidato. La capa de evidencia mantiene el inventario completo y
RAG decide posteriormente cómo fragmentarlo o seleccionarlo. La proyección de
compatibilidad XIAO solo reproduce la forma histórica de un trigger completado; los
comentarios se entregan externamente y no se convierten en evidencia del candidato.

El contrato todavía no está conectado como autoridad general de las rutas
productivas y el baseline no ha sido migrado a él. Sidecars, IDs, manifests y
artefactos RAG conservan sus contratos actuales.

## Estado de detectores alternativos

River 0.26.1 forma parte del runtime reproducible. Page-Hinkley está integrada
técnicamente mediante `PageHinkleyConfig`,
`PageHinkleyAdapter` y la estrategia registrada `page_hinkley`. El adapter recibe
solo `ActivityObservation.value`, propaga literalmente identidad, tiempo y calidad,
y emite `score=None`. Su metadata se limita al contador propio de observaciones,
`min_instances`, `mode` y el estado derivado `warmup_complete`; no lee estado privado
de River. River reinicia el estado algorítmico al recibir la observación posterior a
un drift y el adapter alinea entonces su contador de segmento, sin reset ni cooldown
adicional.

ADWIN también está integrado mediante `ADWINConfig`, `ADWINAdapter` y la estrategia
`adwin`, usando la misma interfaz neutral. `ActivitySignalDefinition.window` sigue
siendo el intervalo físico usado para construir una observación; la ventana
adaptativa de ADWIN es estado interno medido en observaciones. Su metadata pública
incluye `width`, `estimation` y `variance`, pero estas magnitudes no se convierten en
un score común. Ante un drift, ADWIN contrae la ventana en esa actualización; en la
siguiente, River reinicia el detector conforme al lifecycle público de esta versión.

Ambas integraciones están conectadas al runtime cíclico técnico mediante la misma
ruta neutral. El runtime conserva una instancia de señal y una de detector entre
observaciones y ciclos, y persiste envelopes en
`activity_detection_results.jsonl`. La fuente de este plumbing se registra como
`PROVISIONAL_RUNTIME_INPUT`: no resuelve qué vista de preprocessing es adecuada ni
valida metodológicamente la señal o los parámetros. La escala de cada señal deberá
evaluarse antes de calibrar ADWIN. XIAO continúa como
`REFERENCE_DETECTOR` y `REGRESSION_ANCHOR`, no como compromiso de detector final.

La selección de productor también es neutral en el runtime. Un registry explícito
de factories resuelve `ActivitySignalDefinition.signal_id` y crea estado nuevo por
ejecución para las dos familias de señal implementadas:

```text
comment_count_event_window_120s_step_30s
unique_author_count_event_window_120s_step_30s
```

Ambas producen el mismo contrato `ActivityObservation` y pueden alimentar tanto
Page-Hinkley como ADWIN sin branching por combinación. La definición resuelta sigue
siendo autoridad de ventana, cadencia y base temporal; el ID coherente derivado de
esos parámetros continúa resolviendo la misma familia registrada. El paso cíclico conserva
`author_id` como atributo opcional para que la señal de autores aplique su política
documentada; no lo convierte en requisito de la señal de comentarios ni resuelve
la selección futura de preprocessing.

### Backlog posterior a SIGNAL-RUNTIME-REGISTRY-01

- **RESOLVED:** integración técnica de Page-Hinkley; integración técnica de ADWIN;
  `CYCLIC-DETECTION-RUNTIME`; selección runtime de `comment_count` y
  `unique_author_count` mediante configuración.
- **READY WITH MINOR DEBT:** `USABILITY-EXPERIMENT-01` para las señales y detectores
  actualmente implementados.
- **OPEN:** `PREPROCESSING-EVAL`; diseño e integración de señales de polarización;
  semántica de fingerprint experimental y `config_hash`; handoff runtime hacia
  `EventCandidate`; consolidación de configuración de dataset; manifest unificado;
  aislamiento de outputs por `run_id`; duplicación de perfiles.
- **DEFERRED / CONDITIONAL:** KSWIN, solo ante una señal o caso donde el cambio
  distribucional aporte información que Page-Hinkley y ADWIN no representen.
- **DEFERRED:** estudio específico de XIAO; estudio del baseline diario;
  modernización retrospectiva; durabilidad online A9.
