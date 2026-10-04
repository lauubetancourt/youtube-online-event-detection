# Implementacion tecnica de G-2 jerarquico

**Estado:** CURRENT IMPLEMENTATION REFERENCE
**Alcance:** arquitectura y contratos de G-2 jerárquico; no registra resultados
experimentales ni convierte el modo externo en ejecución automática.
**Arquitectura end-to-end:** [pipeline_architecture.md](pipeline_architecture.md).

Este documento describe la implementacion de G-2 jerarquico como fase posterior
y opcional del pipeline. No documenta resultados experimentales.

## Proposito

G-2 jerarquico contrasta evidencia interna de YouTube con evidencia externa de
noticias, manteniendo dos niveles:

```text
event_id
-> video_id
-> context_unit_id
-> comment_id
```

`event_id` conserva la alerta global. `video_id` funciona como subunidad de
validacion externa para evitar mezclar temas, entidades o evidencias entre
videos de un mismo evento.

## Configuracion central

La configuracion se concentra en `RagG2HierarchicalConfig`, dentro de
`youtube_pipeline/rag_generation_g2_hierarchical.py`.

Campos principales:

- `consumer_dir`: carpeta con `rag_validation_inputs.jsonl`,
  `rag_context_payloads.jsonl` y `rag_consumer_manifest.json`.
- `output_dir`: carpeta donde se escribirian artefactos G-2 en modo real.
- `event_id`: evento a ejecutar en modo real. En dry-run puede omitirse para
  planear todos los eventos.
- `query_model` y `validation_model`: modelos para query por video y validacion.
- `provider`: proveedor LLM. El soporte actual es `openai`.
- `temperature`: temperatura solicitada; si el modelo no la soporta, no se envia.
- `serper_url`, `serper_gl`, `serper_hl`, `serper_type`,
  `serper_num_results`: parametros de Serper News.
- `search_days_before` y `search_days_after`: ventana de busqueda alrededor del
  trigger.
- `max_videos_per_event_batch`: limite de videos por evento en el lote actual.
- `max_estimated_tokens_per_event_batch`: limite estimado de tokens por evento
  en el lote actual.
- `max_llm_calls_per_batch`: limite estimado de llamadas LLM por lote.
- `max_serper_calls_per_batch`: limite estimado de llamadas Serper por lote.
- `max_estimated_cost_usd_per_batch`: limite de costo si existen tarifas.
- `params.cost_estimation`: tarifas opcionales para activar el guard de costo.

Si `params.cost_estimation` no define tarifas positivas, el manifest o dry-run
reporta:

```text
cost_guard_status = not_enforced_missing_rates
```

## Modo dry-run

El dry-run construye un plan determinista sin llamadas externas y sin escrituras.
No carga API keys, no llama OpenAI, no llama Serper, no crea embeddings y no crea
vectorstores.

Ejemplo:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_rag_generation_g2_hierarchical.py \
  --consumer-dir experiments/xiao/media/log_3/rag_consumer \
  --output-dir experiments/xiao/media/log_3/rag_generation_g2_hierarchical_batch \
  --dry-run
```

El dry-run muestra:

- eventos que se planearian;
- videos seleccionados para el lote actual;
- videos pendientes como `pending_batch` o `pending_budget_limit`;
- tokens aproximados;
- llamadas LLM estimadas;
- llamadas Serper estimadas;
- previews deterministas de query basados en titulo/canal, sin LLM;
- estado del guard de costo.

## Modo real futuro

El modo real debe ejecutarse solo cuando la fase de evaluacion lo apruebe.
Requiere `event_id`, `OPENAI_API_KEY` y `SERPER_API_KEY`.

Ejemplo de forma de ejecucion futura:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_rag_generation_g2_hierarchical.py \
  --consumer-dir experiments/xiao/media/log_3/rag_consumer \
  --output-dir experiments/xiao/media/log_3/rag_generation_g2_hierarchical_batch \
  --event-id evt_XXXXXXXXXXXX
```

## Artefactos de modo real

En modo real, la fase produciria sidecars nuevos en `output_dir`:

- `rag_video_news_queries.jsonl`
- `rag_video_external_evidence.jsonl`
- `rag_video_validation_reports.jsonl`
- `rag_event_validation_summary.jsonl`
- `rag_generation_manifest.json`
- `rag_raw_model_responses.jsonl`

No reemplaza sidecars RAG, resultados G-1, resultados G-2 previos ni artefactos
del pipeline base.

## Politica de claims

> Contract update: new hierarchical artifacts use the three-state validation
> contract and metadata-first query generation documented in
> `docs/g1_g2_validation_semantics.md`. Legacy artifacts described below remain
> readable, but their additional query/status fields are not generated anew.

Reglas:

- puede ser `null`;
- debe tener `claim_query_status`;
- se genera, si aplica, dentro de la misma llamada futura que produce
  `primary_event_query`;
- no genera llamadas LLM adicionales;
- no se ejecuta;
- no produce reportes propios;
- no activa fact-checking de afirmaciones especificas.

Valores permitidos de `claim_query_status`:

- `not_applicable`
- `no_clear_factual_claim`
- `multiple_claims_no_selection_policy`
- `registered_not_executed`

## Politica batch

El numero de videos no excluye eventos. Los videos se ordenan de forma
determinista por:

1. mayor numero de comentarios asociados al evento;
2. mayor numero de unidades de contexto;
3. menor timestamp del primer comentario;
4. `video_id` como desempate estable.

Los videos no procesados en el lote actual quedan como pendientes, no excluidos.
La sintesis global queda parcial si quedan videos pendientes.

Estados de video:

- `processed`
- `pending_batch`
- `pending_budget_limit`
- `skipped_no_context`
- `failed_retrieval`
- `failed_validation`

Estados de evento:

- `complete`
- `partial_pending_batch`
- `partial_errors`
- `not_started`
- `failed`

## Validaciones internas

La implementacion verifica:

- que cada bundle de video solo contenga unidades y comentarios de ese video;
- que un reporte por video cite solo `comment_id` del mismo `video_id`;
- que cite solo `context_unit_id` del mismo `video_id`;
- que evidencia externa citada pertenezca al mismo `event_id + video_id`;
- que `external_event` requiera cita externa valida;
- que un video sin evidencia externa no se clasifique como `external_event`;
- legacy `claim_verification_query` fields are tolerated when reading historical
  artifacts, but new runs neither request nor persist them.

## Pruebas sin red

Las pruebas unitarias en `tests/test_rag_generation_g2_hierarchical.py` usan
fixtures sinteticos y no llaman OpenAI, Serper, embeddings ni vectorstores.

Comando:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_rag_generation_g2_hierarchical
```

## Lo que G-2 jerarquico no hace todavia

- No ejecuta claim verification.
- No usa embeddings.
- No usa ChromaDB.
- No usa vectorstore.
- No hace query expansion.
- No hace multi-query adicional.
- No usa agentic RAG.
- No usa self-reflective RAG.
- No usa knowledge graphs.
