# Clasificador individual de postura

El clasificador standalone produce el contrato existente `CommentStance`.
Está integrado técnicamente, pero no está validado ni conectado al productor
temporal, a los detectores o a RAG. No clasifica sentimiento ni genera confidence.

## Referencia inspeccionada antes de implementar

La referencia principal es `.agents/examples/CommentClassificationGPT.ipynb`,
SHA-256 `efa55e4b5d29657261c17fc6a8a083cbb3eb1ff05c37833e7284df006367e9e2`.
Se conserva esta elección del contrato previo de clasificación.

El notebook lee CSV con `comment_id`, `text`, `is_reply` y
`reply_to_comment_id`. Da un contexto sobre la reforma pensional colombiana y
una proposición conocida; no descubre proposiciones. El sistema define la tarea
y el usuario aporta el comentario. La escala es: 1 completamente en contra,
2 en contra, 3 neutral, 4 a favor y 5 completamente a favor. También pide un
sentimiento 1–3, excluido de esta implementación.

Un reply recibe el texto del padre inmediato en el system prompt. Su búsqueda
usa `df.loc[...].text.values[0]` sin controlar padres ausentes. No reconstruye
el thread completo. Usa `AsyncOpenAI`, `gpt-4.1-2025-04-14`, `temperature=0`,
`top_p=0.9`, `max_tokens=10`, un semáforo de 15 llamadas, y `asyncio.gather`.
Procesa raíces y luego replies por proposición; conserva `comment_id`.
Reintenta todas las excepciones hasta tres veces con `2**attempt + random.random()`;
el semáforo queda ocupado durante la espera. Fallos y parsing inválido terminan
en `np.nan`. Separa la salida compacta, por ejemplo `4,3`, mediante `split(',')`.
Acumula `usage.total_tokens` y persiste columnas por ID en `comments_class.csv`.
No estima coste monetario ni limita TPM en esta referencia principal.

Existe otro notebook homónimo en `notebooks/CommentClassificationGPT.ipynb`,
SHA-256 `c13f689fb8567f6c4fb8d5b742cccb0e5685bac73017c6be07301512bd8c5dab`.
Esa variante legacy es síncrona, añade contexto de vídeo, usa una estimación de
800 tokens y un límite de 450000 TPM asociado a su tier. Declara seis intentos
con tenacity, pero captura excepciones dentro de la función decorada y devuelve
`np.nan`, evitando que muchos errores lleguen al retry. No se mezcla ese flujo
con la referencia asíncrona ni se asumen sus límites de cuenta.

## Adaptación al pipeline

| REFERENCE_BEHAVIOR | PIPELINE_CONSTRAINT | ADAPTATION_DECISION |
| --- | --- | --- |
| Postura y sentimiento en texto CSV | Stance-only, tipado y validable | JSON Schema estricto `status`, `stance`; sin sentimiento |
| Toda respuesta intenta producir 1–5 | Neutral no equivale a ausencia de opinión | `NOT_APPLICABLE` y `UNCERTAIN` llevan stance null |
| Padre sin guarda, incluido en system | Contexto causal y texto no confiable | Padre inmediato con fecha ≤ reply; texto de padre/reply en user JSON |
| Padre ausente provoca fallo | Ausencia explícita, sin inventar contexto | `SKIPPED`, razón `missing_parent`, ninguna llamada |
| Semáforo fijo de 15 | Límites de cuenta desconocidos | Concurrencia configurable, default técnico 1; espera fuera del semáforo |
| Retry para cualquier excepción | Separar fallos transitorios de salida inválida | Tres intentos; solo fallos transitorios; salida inválida no se reintenta |
| Diez tokens para pareja numérica | JSON necesita otro presupuesto | `max_completion_tokens=128`, DEVELOPMENT_DEFAULT, no calibrado |
| Solo tokens totales, fallo contabilizado como cero | Consumo desconocido no equivale a cero | Tokens input/output/total reportados y número de intentos sin usage |
| CSV actualizado por ID | Artefactos y configuración trazables | JSONL, manifest y resolved_config; guardas de duplicados y sobrescritura |

Se mantiene el snapshot del modelo. Su ficha oficial confirma soporte de
[Structured Outputs](https://developers.openai.com/api/docs/models/gpt-4.1).
La API usa [JSON Schema estricto](https://developers.openai.com/api/docs/guides/structured-outputs)
y una validación local adicional exige stance 1–5 exclusivamente para CLASSIFIED.
Rechaza campos extra, claves repetidas, bool como categoría, estados desconocidos
y combinaciones inconsistentes. Refusal/content_filter produce SKIPPED;
salida incompleta o malformada produce ERROR.

## Contratos y configuración

`CommentStanceClassifier.classify(comment)` recibe `StanceCommentInput` y retorna
`CommentStance`. Una `StanceProposition` explícita queda fijada por configuración
para toda la instancia. El batch puede resolver padres por ID usando únicamente
su texto inmediato, sin necesitar la clasificación previa del padre.
El resultado del batch conserva el orden de entrada; la persistencia ordena por
`event_time_utc`, luego `comment_id`, sin depender del orden de completion.

Ejemplo mínimo de perfil standalone:

```json
{
  "identity": {"run_id": "stance_development_01"},
  "polarization": {
    "classifier": {
      "proposition": {
        "proposition_id": "provided_proposition",
        "proposition_text": "Texto de la proposición dada",
        "context": "Contexto auxiliar proporcionado",
        "context_version": "provided_v1"
      },
      "model": "gpt-4.1-2025-04-14",
      "prompt_id": "comment_stance_likert5",
      "prompt_version": "1",
      "max_concurrency": 1
    }
  }
}
```

También se configuran retry_policy, temperature, top_p, max_completion_tokens,
request_timeout_seconds y el guard opcional max_input_characters. Este último
cuenta caracteres del prompt construido: no estima tokens ni trunca el texto.
Los tipos son frozen/slots; se rechazan claves desconocidas, números no finitos,
tipos inválidos y bool como número. Ningún secreto entra en RunConfig o su hash.
Las configuraciones anteriores de medidas aisladas y señales siguen funcionando.

El input JSONL contiene comment_id, author_id nullable, event_time_utc UTC, text,
y opcionalmente parent_comment_id, parent_text, parent_event_time_utc e is_deleted.
Un padre explícito exige identidad y fecha para comprobar causalidad.
No se admiten IDs duplicados dentro de una ejecución. Un comentario vacío o
eliminado se omite sin llamada. Un autor ausente permite clasificación y marca
`degraded_missing_author`; la agregación por autor decidirá después su exclusión.

## Ejecución y artefactos

`scripts/run_comment_stance_classifier.py --config perfil.json --input-jsonl input.jsonl`
solo valida/prepara por defecto. `--execute --output-root destino_nuevo` habilita
inferencia explícitamente; esa ejecución puede generar cargos. No se realizaron
llamadas reales en esta fase. `--run-id` es un override operativo disponible.

Se reutilizan requests/python-dotenv ya aprobados; no se instala el SDK OpenAI.
El transporte HTTP llama Chat Completions vía `asyncio.to_thread`, con timeout,
`store=false` y sin redirects. Lee OPENAI_API_KEY del entorno o .env solo cuando
se invoca. No persiste credenciales, texto del error HTTP ni respuestas crudas.

`comment_stances.jsonl` conserva los campos de CommentStance y un envelope con
parent_comment_id, quality_reasons, attempts, classification_time_utc, input_hash,
error_reason saneado y model_response_id. `classification_status` es el nombre
existente del contrato, no un nuevo campo status paralelo.

`classifier_manifest.json` gobierna classifier_run_id, modelo/proveedor, prompt
versionado/hash/schema, proposición/contexto/version/hash, concurrencia, retries,
parámetros efectivos, tokens agregados, conteos por estado y referencias/hash
del artefacto. `resolved_config.json` conserva la configuración canónica y su
hash vigente; no se rediseña la participación de run_id.

HTTP 408/409/429/5xx, timeout y conexión son reintentables. Otros HTTP fallan
sin retry. Backoff exponencial con jitter y Retry-After numérico se limita por
max_delay_seconds. No se inventa un límite TPM universal ni un precio fijo.
La concurrencia, presupuesto de salida y número de intentos están acotados;
no se garantiza un techo monetario total. Tokens desconocidos se señalan mediante
unreported_attempts/usage_complete=false; los totales solo suman usage recibido,
incluido el de respuestas inválidas. monetary_cost=null, NOT_ESTIMATED.

Las tres rutas de salida se crean exclusivamente antes de inferencia. No se
sobrescriben corridas existentes. Una interrupción inesperada puede dejar archivos
parciales: no se promete persistencia atómica, reanudación ni idempotencia A9.

## Backlog

- RESOLVED: integración técnica standalone del clasificador y persistencia trazable.
- NEXT: POLARIZATION-CLASSIFIER-VALIDATION; la precisión no se deduce de mocks.
- OPEN: proposition discovery, runtime real de polarización, validación/calibración,
  preprocessing, detector gap/reset policy, asociación/fusión y deudas de config/hash.
- OPEN operativo: evaluación de throughput/límites de cuenta y coste opcional;
  no se hereda el tier del notebook legacy.
- DEFERRED: A9 y durabilidad cross-restart. No cambia actividad/detectores/RAG.
