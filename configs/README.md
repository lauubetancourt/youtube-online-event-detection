# Configuración de ejecuciones

Esta guía amplía el [paso de configuración del README](../README.md#2-preparar-y-configurar-el-análisis). Un perfil JSON de `RunConfig` indica qué datos, tema y parámetros utilizará una ejecución. Al cargarlo, el programa completa los valores por defecto, comprueba los campos y resuelve las rutas. Al ejecutar el análisis, guarda la configuración efectiva en `resolved_config.json` y calcula un hash para identificarla.

Las claves de API se proporcionan mediante variables de entorno y no se escriben en el perfil. Comience con [`examples/reproduccion_cronologica.example.json`](examples/reproduccion_cronologica.example.json): cópielo en `configs/local/`, adapte sus valores y conserve el archivo de ejemplo como referencia. El perfil local no se registra en Git.

## Recorrido integrado

Para reproducir los cortes temporales con `scripts/run_causal_end_to_end.py`, el perfil reúne estas secciones:

| Sección | Función |
|---|---|
| `identity.run_id` | Identificador de la ejecución. |
| `topic_profile` | Tema y proposición de clasificación. |
| `data.local_files` | Archivos locales de videos y comentarios. |
| `data.experimental_input` | Política `causal_minimal_v1` de preparación causal. |
| `simulation.causal_end_to_end` | Inicio, fin y paso del reloj simulado, directorio de salida y archivos de posturas ya clasificadas. |
| `signals.activity` y `signals.polarization` | Métricas o medidas, ventanas y cadencias independientes. |
| `detection.routes` y configuración del detector | Detector que se aplicará a cada señal seleccionada. |
| `association` | Política y tolerancia de asociación temporal. |
| `rag.candidate_evidence` y `rag.candidate_validation` | Evidencia interna y validaciones G-1/G-2. |
| `artifacts` | Forma de registrar los resultados y su trazabilidad. |

El mismo perfil puede utilizarse primero para clasificar posturas y después para la reproducción. En el comando del clasificador, `--run-id` debe coincidir con `signals.polarization.classification_run_id`. Este identificador distingue la clasificación de `identity.run_id`, que nombra la reproducción. El ejemplo incluye `topic_profile` dentro del mismo JSON para que pueda definir allí el tema, la proposición y su contexto. En `simulation.causal_end_to_end.stance_artifact_path` y `stance_manifest_path`, indique dónde quedarán los dos archivos producidos por el clasificador. El programa comprueba que las posturas correspondan al mismo corpus y tema.

Las rutas de `data.local_files.comments_path`, `rag.candidate_evidence.source_path` y los archivos de posturas deben corresponder al conjunto que desea analizar. Si usa rutas relativas, ejecute los comandos desde la raíz del repositorio con `--base-dir .`. Antes de añadir `--execute`, el programa muestra el identificador y el hash de la ejecución, el intervalo, el paso, el directorio de salida y si se ejecutarían G-1 y G-2. Revise estos datos para comprobar que está usando el perfil previsto.

## Antes de ejecutar

En el ejemplo, cambie las rutas de comentarios y videos, las fechas del reloj y del cálculo de polarización, el tema, los identificadores y las rutas de salida. Revise también los parámetros de señales, detectores, asociación y evidencia para su caso. Los valores marcados como `TEST_ONLY` son ilustrativos. La comprobación del plan de reproducción verifica la configuración, pero no confirma que los archivos existan o que las posturas ya estén clasificadas.

Conserve el perfil, `resolved_config.json`, los archivos de entrada y `classifier_manifest.json` si necesita revisar cómo se obtuvo un resultado. La [guía de entrada y reproducción](../docs/causal_end_to_end_replay.md) explica los formatos admitidos y los límites de este recorrido.
