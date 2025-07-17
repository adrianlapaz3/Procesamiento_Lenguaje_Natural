# Desafio 2
## Word Embeddings para *Hamlet* de Shakespeare
El objetivo principal de este proyecto es explorar las relaciones semánticas dentro del texto de *Hamlet* aplicando *word2vec*, una popular técnica de *word embeddings*. El *notebook* guía a través de los siguientes pasos:
1.  **Extracción y preprocesamiento de texto**: aislamiento del diálogo de la versión del corpus *gutenberg* de "Hamlet", limpieza del texto y aplicación de tokenización, lematización y eliminación de *stopwords*.
2.  **Entrenamiento del modelo *word2vec***: entrenamiento de un modelo *word2vec* *skip-gram* utilizando la librería *gensim* para aprender representaciones vectoriales densas (*embeddings*) para palabras basándose en sus patrones de co-ocurrencia.
3.  **Análisis semántico**: evaluación de los *embeddings* aprendidos mediante consultas de similitud de palabras y pruebas de analogía.
4.  **Reducción de dimensionalidad y visualización**: uso de *t-sne* para reducir los vectores de palabras de alta dimensión a 2d y 3d para la exploración visual de clústeres y relaciones de palabras.

## Características

* **Descarga automática de recursos *nltk***: asegura que todos los corpus y modelos *nltk* necesarios estén disponibles.
* **Preprocesamiento de texto robusto**: incluye limpieza basada en expresiones regulares para eliminar elementos estructurales (marcadores de acto/escena, etiquetas de hablantes, direcciones de escena) y pasos estándar de *pln* (tokenización, minúsculas, lematización, eliminación de *stopwords*).
* **Entrenamiento de *word2vec* con seguimiento de pérdidas**: implementa una función de *callback* personalizada para monitorear la pérdida de entrenamiento por época, lo que proporciona información sobre la convergencia del modelo.
* **Consultas de similitud de palabras**: demuestra la funcionalidad `most_similar` para encontrar palabras semánticamente cercanas a términos de interés dentro del contexto de "Hamlet".
* **Aritmética vectorial para analogías**: intenta realizar analogías vectoriales (por ejemplo, "*rey*" - "*hombre*" + "*mujer*" = "*reina*") para probar la capacidad del modelo para capturar relaciones semánticas abstractas.
* **Visualizaciones interactivas *t-sne***: genera diagramas de dispersión interactivos en 2d y 3d utilizando *plotly*, lo que permite la exploración visual de clústeres de palabras en el espacio de *embedding*.


## Análisis y resultados

### Observaciones del preprocesamiento

El *pipeline* de preprocesamiento extrajo aproximadamente 3735 oraciones efectivas de "Hamlet" para el entrenamiento de *word2vec*. Las observaciones clave incluyen:
* **Inclusión de elementos estructurales**: el preprocesamiento retuvo términos como `actus`, `primus`, `scoena`, `prima`, `enter`, `barnardo`, `francisco`, y abreviaturas de nombres de personajes (`fran`). Esto es útil para que el modelo aprenda el contexto de estos elementos y para analizar interacciones de personajes.
* **Ortografías arcaicas**: el modelo aprendió *embeddings* para ortografías arcaicas (ej., `vnfold`, `liue`), reflejando el vocabulario específico del inglés shakespeariano.
* **Limpieza efectiva**: la lematización y la eliminación de *stopwords* mejoraron significativamente la calidad de los *tokens* al reducir las variaciones de palabras y eliminar palabras comunes menos informativas.

### Entrenamiento del modelo *word2vec*

El modelo *word2vec* fue entrenado con 3735 oraciones procesadas durante 100 épocas. La pérdida por época disminuyó progresivamente de ~274k a ~28k, lo que indica un aprendizaje efectivo y la convergencia del modelo. El tamaño del vocabulario final fue de 4145 palabras, utilizando `min_count=1`, lo que implica que incluso palabras muy raras tienen *embeddings*, aunque su calidad puede ser menor debido a datos contextuales limitados.

### Pruebas de similitud y analogía

#### Palabras más similares
Los resultados de `most_similar` demuestran la capacidad del modelo para capturar asociaciones semánticas muy específicas y contextualmente relevantes dentro de "Hamlet" para las palabras presentes en su vocabulario.
* Para `ophelia`, se observan asociaciones como `beautifed`, `idoll`, `orizons` y `nimph` (similitudes >0.73), que reflejan con precisión su caracterización y las descripciones textuales.
* La fuerte relación de `queen` con `willow` y `aslant` (ambas >0.77) es particularmente perspicaz, ya que vincula directamente a la descripción icónica de la muerte de *ophelia* narrada por *gertrudis*.
* `ghost` muestra una conexión temática con `adulterate` (0.6547).
* Términos geopolíticos como `denmark` y `fortinbras` se asocian con un léxico militar y político relevante (`polake`, `warres`, `prison` para `denmark`; `compact`, `slay`, `inheritance` para `fortinbras`), con similitudes a menudo superiores a 0.79.
* El modelo también aprendió asociaciones para ortografías arcaicas como `discouery` (similar a `king`).
* Sin embargo, se identificaron asociaciones menos intuitivas, por ejemplo, para `hamlet` (`vnbrac`, `doublet`), que podrían interpretarse como ruido estadístico o la influencia de co-ocurrencias esporádicas en un corpus reducido, con puntuaciones de similitud moderadas (0.53-0.58).

#### Similitudes entre pares
La similitud coseno entre pares seleccionados cuantifica su cercanía en el espacio vectorial aprendido:
* La relación `king` - `claudius` (0.4284) indica una asociación semántica moderada, reflejando su identidad parcialmente superpuesta en la narrativa.
* Pares como `hamlet` - `ophelia` (0.2943) y `king` - `queen` (0.2157) exhiben similitudes más débiles, lo que podría sugerir que sus perfiles distribucionales no son suficientemente intercambiables en el corpus.
* La similitud `death` - `ghost` (0.2938) es también moderada-baja, posiblemente indicando que, aunque conceptualmente ligados, sus contextos de uso en "hamlet" son suficientemente distintos.

#### Limitaciones del vocabulario
Una limitación crítica observada es la ausencia de términos temáticos clave como `love`, `madness`, `revenge`, `skull` y `poison` en el vocabulario del modelo. Esto es probable debido a la frecuencia de los términos o al preprocesamiento, lo que impide una representación semántica completa de dimensiones narrativas fundamentales de "Hamlet".

#### Pruebas de analogía
Las pruebas de analogía (`king` - `man` + `woman` ≈ `queen` y `laertes` - `polonius` + `ghost` ≈ `hamlet`) no identificaron los términos canónicos esperados. Los vectores resultantes de las operaciones aritméticas mostraron bajas similitudes coseno (0.3-0.5) con los términos más cercanos del vocabulario. Este resultado se atribuye principalmente a:
* **Especificidad del corpus**: El modelo fue entrenado exclusivamente en "Hamlet", un corpus relativamente pequeño y altamente especializado. Esto limita su capacidad para aprender subestructuras lineales generalizables (desplazamientos vectoriales) que representan relaciones semánticas abstractas (como género o roles familiares) que se encuentran comúnmente en corpus más grandes y diversos.
* **Falta de relaciones generalizables**: Los *embeddings* capturan principalmente patrones de co-ocurrencia específicos de la narrativa en lugar de relaciones lingüísticas más amplias.

### Visualizaciones *t-sne*

Las visualizaciones *t-sne* (*t-distributed stochastic neighbor embedding*) son una técnica de reducción de dimensionalidad no lineal utilizada para representar relaciones de similitud semántica entre palabras.

#### Gráfico *t-sne* 2d
La proyección 2d muestra una organización general útil para captar agrupaciones temáticas, aunque con cierto solapamiento en regiones de alta densidad. Una agrupación densa en la parte superior derecha incluye a personajes prominentes (`hamlet`, `claudius`, `ghost`, `horatio`, `king`, `queen`, `gertrude`, `polonius`), resaltando su prominencia en contextos comunes. Sin embargo, la bidimensionalidad impone limitaciones: palabras clave como `himselfe`, `ere`, `grace` o `sword` aparecen más aisladas, y las conexiones entre capas semánticas intermedias son difíciles de observar.

#### Gráfico *t-sne* 3d
Las visualizaciones tridimensionales ofrecen una representación más rica y matizada del espacio semántico. En el gráfico 3d, los clústeres observados en la vista 2d se reafirman, pero se organizan de manera más coherente, a menudo formando una estructura esférica clara que indica una fuerte cohesión semántica local. Además, se revelan subestructuras no evidentes en 2d: términos como `madness`, `death`, `revenge` y `father` parecen formar un subclúster temático que representa la trama psicológica de la obra. Palabras que parecían aisladas en la vista 2d se integran de manera más natural, sugiriendo que su aparente separación era un artefacto de la reducción bidimensional. En conjunto, las vistas 3d no solo confirman agrupaciones semánticas esperadas, sino que revelan gradientes y transiciones contextuales que aportan una capa adicional de interpretabilidad.

## Dependencias

Las principales dependencias utilizadas en este proyecto son:

* **Python 3.x**
* **NLTK**: para tokenización de texto, lematización y *stopwords*.
* **Gensim**: para el entrenamiento del modelo *word2vec*.
* **NumPy**: para operaciones numéricas, especialmente con vectores.
* **Pandas**: para manipulación de datos (aunque no se usa intensamente para operaciones de *dataframe*, a menudo se usa implícitamente por otras librerías).
* **Scikit-learn**: específicamente para `tsne` para la reducción de dimensionalidad.
* **Plotly**: para visualizaciones de datos interactivas (diagramas de dispersión 2d y 3d).
* **TensorFlow** y **Keras**: aunque listadas en los comandos iniciales de `pip`, no se usan directamente para el modelo *word2vec* en sí, pero pueden estar presentes en el entorno de *colab* y su compatibilidad con *numpy* se gestiona.
