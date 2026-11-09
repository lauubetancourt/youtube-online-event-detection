# Detección de eventos en YouTube

Este repositorio contiene un prototipo para la detección de eventos en YouTube. Su enfoque de detección en línea combina señales de actividad y medidas de polarización para detectar cambios estadísticos y generar eventos candidatos con la información disponible en cada corte temporal. Después examina esos candidatos mediante dos mecanismos que emplean modelos de lenguaje: la evaluación del patrón conversacional (G-1) y el contraste con noticias y otras fuentes públicas (G-2).

Los pasos siguientes explican cómo obtener o preparar los datos, configurar el análisis y ejecutar una reproducción cronológica en su equipo.

## Requisitos

Necesita Python 3.14.3, indicado en `.python-version`. Desde la raíz del repositorio, cree un entorno virtual e instale las dependencias:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copie el archivo de ejemplo y complete solo las claves de los servicios que vaya a utilizar:

```bash
cp .env.example .env
```

Obtenga `YOUTUBE_API_KEY` en [Google Cloud](https://developers.google.com/youtube/data/guides/authentication), tras habilitar YouTube Data API v3; `OPENAI_API_KEY` en la [plataforma de OpenAI](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key); y `SERPER_API_KEY` en [Serper](https://serper.dev/). La primera se usa para extraer datos, la segunda para clasificar posturas y validar candidatos, y la tercera para buscar fuentes públicas. El archivo `.env` es local y está excluido de Git.

## 1. Elegir u obtener los datos

Puede partir de archivos propios de comentarios y videos si cumplen los [requisitos de entrada](docs/causal_end_to_end_replay.md#entrada-local). Si necesita obtenerlos de YouTube, ejecute una extracción:

```bash
python -m youtube_pipeline.run_pipeline extract \
  --query 'TEMA' --published-after 'AAAA-MM-DDT00:00:00Z' \
  --published-before 'AAAA-MM-DDT00:00:00Z' \
  --min-views 0 --min-comments 0 \
  --data-root data/mi_extraccion --save-legacy-csv
```

Reemplace `TEMA` por los términos de búsqueda y las fechas de ejemplo por el inicio y el fin del período que desea consultar. El comando busca videos y extrae sus comentarios. Los mínimos de vistas y comentarios están puestos en cero para que el ejemplo no descarte videos por esos criterios. Si desea filtrarlos, cambie esos valores. Los filtros se aplican después de la búsqueda. La extracción conserva además topes por defecto de 500 videos y 5 000 comentarios por video, ajustables con `--max-results` y `--max-comments`. Consulte las demás opciones con `python -m youtube_pipeline.run_pipeline extract --help`.

Al terminar, encontrará `data/mi_extraccion/comments.csv` y `data/mi_extraccion/videos_preliminares.csv`, además de los archivos Bronze (JSONL) y Silver (Parquet). Los dos CSV sirven como entrada para los pasos siguientes. Si realiza otra extracción, use otro `--data-root` para conservar los archivos de cada lectura por separado.

## 2. Preparar y configurar el análisis

El análisis utiliza el CSV de comentarios y el archivo de videos de la misma extracción. Si parte de archivos propios, compruebe que los comentarios tengan los campos necesarios y que sus `video_id` correspondan a los del archivo de videos. La [guía de entrada](docs/causal_end_to_end_replay.md#entrada-local) indica las columnas y formatos admitidos.

Copie el [perfil de ejemplo](configs/examples/reproduccion_cronologica.example.json) para crear su configuración local:

```bash
mkdir -p configs/local
cp configs/examples/reproduccion_cronologica.example.json configs/local/mi_analisis.json
```

En `configs/local/mi_analisis.json`, ajuste el tema y su proposición, las rutas de los datos y los tiempos de inicio y fin. El inicio de la reproducción y el de polarización deben quedar antes del primer comentario del CSV; ambos finales, después del último. El clasificador procesa todo el CSV y la reproducción exige las posturas de ese mismo conjunto. Si desea analizar solo un período, prepare primero un CSV con los comentarios de ese período y úselo tanto para clasificar como para reproducir. Revise también ventanas, cadencias, detectores y rutas de salida: los valores del ejemplo son ilustrativos, no una configuración recomendada. El perfil local está excluido de Git. La extracción no lo crea automáticamente. La [guía de configuración](configs/README.md) explica cómo se relacionan sus campos.

La reproducción necesita posturas clasificadas. Puede obtenerlas con el clasificador del repositorio usando el mismo perfil. Indique en `--run-id` el valor que haya puesto en `signals.polarization.classification_run_id` y use el mismo CSV que figura en `data.local_files.comments_path`. Revise primero el plan y después ejecute:

```bash
python scripts/run_comment_stance_classifier.py \
  --config configs/local/mi_analisis.json --base-dir . \
  --run-id ejemplo_posturas --comments-path data/mi_extraccion/comments.csv \
  --output-root outputs/stances/ejemplo_posturas --dry-run
python scripts/run_comment_stance_classifier.py \
  --config configs/local/mi_analisis.json --base-dir . \
  --run-id ejemplo_posturas --comments-path data/mi_extraccion/comments.csv \
  --output-root outputs/stances/ejemplo_posturas --execute
```

Los comandos usan las rutas e identificadores del ejemplo. Si los cambia en el perfil, cámbielos también aquí. La primera orden solo muestra el plan. La segunda consulta al modelo y guarda `comment_stances.jsonl` y `classifier_manifest.json` en `outputs/stances/ejemplo_posturas`.

En el perfil, `simulation.causal_end_to_end.stance_artifact_path` y `stance_manifest_path` deben apuntar a esos dos archivos. El programa comprueba que correspondan al CSV, al tema y al identificador de clasificación configurados. Las posturas se calculan **antes** de la reproducción, pero solo entran al análisis cuando se incorpora su comentario.

## 3. Reproducir los cortes temporales

Cuando estén listos los archivos, las posturas y el perfil, revise el plan de reproducción:

```bash
python scripts/run_causal_end_to_end.py --config configs/local/mi_analisis.json --base-dir .
```

Este comando muestra el intervalo, el paso del reloj simulado y el directorio de salida, sin procesar comentarios ni llamar a servicios externos. Si el plan es el esperado, elija una de estas dos formas de ejecutar el análisis:

```bash
# Construir señales, alertas, candidatos y evidencia interna:
python scripts/run_causal_end_to_end.py --config configs/local/mi_analisis.json --base-dir . \
  --execute --skip-rag

# Incluir también G-1 y G-2:
python scripts/run_causal_end_to_end.py --config configs/local/mi_analisis.json --base-dir . --execute
```

La opción `--skip-rag` termina después de construir la evidencia interna. La otra también ejecuta G-1 y G-2 cuando hay evidencia para evaluarlos y requiere acceso a los servicios indicados en el perfil. Cada ejecución nueva necesita un directorio de salida vacío. Si desea probar las dos opciones, haga una copia del perfil y asigne a cada ejecución un `identity.run_id` y un `simulation.causal_end_to_end.output_dir` diferentes.

Si una ejecución se interrumpe, puede repetir su comando con `--resume` para continuar con la misma configuración y los mismos datos. En el directorio de salida encontrará `resolved_config.json`, `causal_replay_manifest.json` y los resultados de las etapas ejecutadas. La [guía de reproducción](docs/causal_end_to_end_replay.md) explica cómo consultar esos archivos.

La reproducción incorpora en cortes sucesivos los comentarios que ya estaban en el conjunto preparado. No consulta de nuevo la API de YouTube ni establece una latencia máxima de detección.

## Guías de consulta

- [Configuración de ejecuciones](configs/README.md): explica las secciones de los perfiles JSON y cómo se relacionan la clasificación y la reproducción.
- [Entrada y reproducción cronológica](docs/causal_end_to_end_replay.md): detalla los archivos de entrada, las salidas y los límites del recorrido.
