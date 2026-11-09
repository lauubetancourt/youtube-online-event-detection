# Datos de entrada y reproducción cronológica

Esta guía amplía los [pasos de preparación y ejecución del README](../README.md). El prototipo trabaja con comentarios ya recolectados y los incorpora según avanza un reloj simulado. En cada corte entran los registros que cumplen `event_time_utc < data_cutoff_utc`. Los publicados exactamente en el instante del corte aún no entran. Con los comentarios disponibles se actualizan las señales de actividad y polarización, se ejecutan los detectores y se asocian las alertas. Después se reúne la evidencia de los candidatos y, si se eligió la ejecución completa, pueden evaluarse mediante G-1 y G-2.

## Entrada local

Indique el archivo de comentarios en `data.local_files.comments_path`. La reproducción admite CSV, un archivo Parquet o un directorio Parquet. Si va a clasificar las posturas con el script del repositorio, use **CSV**: el clasificador lee ese formato y la reproducción comprueba que las posturas procedan del mismo archivo. Parquet es una opción cuando ya cuenta con un archivo de posturas compatible.

Cada comentario necesita `comment_id` y `video_id`, texto en `text` u `original_text`, y una fecha con zona horaria en `event_time_utc` o `published_at`. Si solo existe `published_at`, el programa obtiene `event_time_utc` durante la lectura. `author_id` permite contar autores y construir distribuciones de opinión. Cuando hay respuestas, `is_reply` y `reply_to_comment_id` o `parent_comment_id` conservan su relación con el comentario padre.

Mantenga identificadores estables. El programa ordena los registros por `(event_time_utc, comment_id)` y elimina repeticiones exactas de un comentario. Si encuentra el mismo `comment_id` con contenido diferente, detiene la ejecución para evitar mezclar registros incompatibles.

Indique los metadatos de los videos en `data.local_files.videos_path` y compruebe que correspondan a los comentarios elegidos. Estos datos se usan para construir la evidencia y formular consultas externas. La extracción puede guardar los CSV de comentarios y videos si se ejecuta con `--save-legacy-csv`. Guarde cada extracción en un directorio distinto: los CSV se escriben con nombres fijos dentro del directorio elegido. La extracción no crea el perfil de reproducción ni clasifica las posturas.

## Posturas y configuración

La reproducción utiliza posturas clasificadas de antemano, con `simulation.causal_end_to_end.stance_mode = precomputed_stances_v1`. Siga el [paso 2 del README](../README.md#2-preparar-y-configurar-el-análisis) para ejecutar el clasificador con el mismo perfil y CSV, y vincular sus archivos de salida. El identificador `signals.polarization.classification_run_id` del perfil debe coincidir con el `--run-id` utilizado al clasificar.

Antes de utilizarlas, el programa comprueba que las posturas correspondan al corpus, los comentarios, los autores, el tema, el modelo y la versión de las instrucciones. Aunque se calcularon antes de iniciar la reproducción, cada postura solo pasa a estar disponible cuando entra su comentario. Así se conserva el orden temporal del análisis.

El perfil también define `data.experimental_input.policy_id = causal_minimal_v1`, el intervalo y el paso del reloj en `simulation.causal_end_to_end`, las ventanas y cadencias de ambas dimensiones, los detectores, la asociación y la evidencia. El paso del reloj y las dos cadencias se configuran de manera independiente. La [guía de configuración](../configs/README.md) describe las secciones del perfil y enlaza un ejemplo que puede adaptar a sus datos.

## Ejecución y resultados

Los comandos para revisar el plan y elegir una modalidad de ejecución están en el [paso 3 del README](../README.md#3-reproducir-los-cortes-temporales). Sin `--execute`, el programa solo muestra el plan. Con `--execute --skip-rag`, procesa los comentarios hasta construir la evidencia interna. Con `--execute` también puede ejecutar G-1 y G-2, según la configuración y la evidencia disponible.

Si una ejecución se interrumpe, añada `--resume` al mismo comando `--execute`. El programa exige la misma configuración resuelta, reutiliza los candidatos completados y vuelve a intentar los resultados parciales o con error. Para una ejecución nueva, el directorio de salida debe estar vacío.

En el directorio de salida encontrará `resolved_config.json`, que recoge los parámetros efectivos, y `causal_replay_manifest.json`, que registra recuentos y rutas de los resultados. También se guardan observaciones, alertas, asociaciones, candidatos y paquetes de evidencia. La ejecución completa añade los resultados de G-1 y G-2 cuando corresponde. G-1 examina el patrón conversacional. G-2 busca y valora fuentes públicas por video y reúne esas valoraciones por candidato.

`CONFIRMED`, `INSUFFICIENT_EVIDENCE` y `NOT_CONFIRMED` son decisiones de validación. Los estados pendiente, fallido o parcial describen el avance de la ejecución y no equivalen a una decisión negativa.

## Qué demuestra esta reproducción

La reproducción muestra cómo avanzan las señales, los detectores y los candidatos al incorporar progresivamente un corpus ya recolectado. Puede reanudarse con la **misma entrada**, pero no incorpora automáticamente una segunda extracción con comentarios nuevos. Por eso, la reanudación no demuestra continuidad entre lecturas distintas de la API.

Las fuentes públicas se filtran por su fecha de publicación. Esa fecha permite aplicar el corte temporal configurado, pero no demuestra cuándo el buscador incorporó cada página a sus resultados.
