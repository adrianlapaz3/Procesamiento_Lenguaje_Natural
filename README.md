## ADRIÁN LAPAZ (1706)
# Desafío 1: calsificación de texto

Este desafio aborda un desafío de Procesamiento del Lenguaje Natural (PLN) utilizando el conjunto de datos **20 Newsgroups** desde *scikit-learn*. El objetivo es explorar técnicas de vectorización, clasificación de texto y análisis de similaridad semántica entre palabras.

## Descripción del desafío

El desafío se divide en tres partes principales:
1.  **Similaridad de documentos**: vectorizar el *corpus* de texto y analizar la similaridad del coseno entre documentos para evaluar la coherencia temática.
2.  **Clasificación de texto**: entrenar y optimizar modelos de clasificación Naïve Bayes (*MultinomialNB* y *ComplementNB*) para predecir la categoría de un documento, maximizando la métrica *f1-score macro*.
3.  **Similaridad de palabras**: transponer la matriz documento-término para crear vectores de palabras y analizar las relaciones semánticas entre términos seleccionados manualmente.


## Resultados principales

### 1. Similaridad entre documentos
Se seleccionaron 5 documentos al azar y se calcularon sus 5 vecinos más similares usando la similaridad del coseno.

- **Coherencia temática**: se observó que los documentos con mayor similaridad a menudo pertenecen a la misma categoría, por ejemplo *rec.sport.hockey* y *rec.sport.baseball* tuvieron una similitud de coseno de 0.37.
- **Restulados**: los valores de similaridad de coseno fueron moderados o bajos (entre 0.14 y 0.37). Esto sugiere que, si bien la similaridad del coseno puede agrupar temas, podría no ser el método más robusto para una clasificación precisa por sí solo.

### 2. Clasificación con Naïve Bayes
Se utilizó **optimización bayesiana (*BayesSearchCV*)** para encontrar los mejores hiperparámetros tanto para el vectorizador *TfidfVectorizer* como para los clasificadores *MultinomialNB* y *ComplementNB*.

- **Rendimiento en entrenamiento (CV, Cross-Validation)**:
  - **MultinomialNB**: mejor *F1-score* (CV) de **0.7626**.
  - **ComplementNB**: mejor *F1-score* (CV) de **0.7661**.
- **Rendimiento en los datos de testeo**:
  - **MultinomialNB**: *F1-score* en Test de **0.6876**.
  - **ComplementNB**: *F1-score* en Test de **0.6969**.

Ambos modelos mostraron un rendimiento muy similar, aunque *ComplementNB* fue apenas superior. El análisis de hiperparámetros reveló que *ComplementNB* logró su mejor rendimiento con un filtrado dinámico de vocabulario y un suavizado mayor, sugiriendo una mayor robustez frente al ruido léxico.

### 3. Similaridad entre palabras
Se analizó la similaridad entre 5 palabras seleccionadas (*ball*, *doctor*, *python*, *space*, *water*) tras transponer la matriz *TF-IDF*.

- **Captura de contexto temático y/o semántica**: el análisis demostró la capacidad del modelo para identificar relaciones contextuales muy específicas.
  - La asociación más fuerte fue entre **python** y **monty** (similaridad de coseno de **0.7138**), una clara referencia semántica al grupo de comedia *"Monty Python"*.
  - Se encontraron fuertes agrupaciones temáticas, como **doctor** con **receptionist** (0.4392) y **space** con **nasa** (0.3304).
  - El modelo probó ser altamente dependiente del contexto del corpus: la palabra *water* no se relacionó con la naturaleza, sino con infraestructura urbana (*towers*, *dpw*, *croton*), reflejando los temas de discusión en los datos.
  
  La técnica de transponer la matriz fue muy efectiva para descubrir conexiones temáticas, semánticas y contextuales entre las palabras, ofreciendo una visión profunda de cómo se utilizan las palabras dentro del conjunto de datos.

---

## Metodología y herramientas

- **Librerías principales**: *scikit-learn*, *numpy*, *skopt*.
- **Vectorización**: *TfidfVectorizer*.
- **Modelos**: *MultinomialNB*, *ComplementNB*.
- **Métrica de evaluación**: *f1_score* (*macro average*).
- **Técnica de optimización**: búsqueda bayesiana (*BayesSearchCV*) para una sintonización eficiente de hiperparámetros.

## Conclusión
La **similaridad del coseno** es útil para la exploración temática, pero los modelos de clasificación como **Naïve Bayes** son superiores para tareas de predicción. Además, el análisis de similaridad de palabras sobre la matriz transpuesta reveló ser una técnica muy poderosa para descubrir **relaciones semánticas** en el texto.

---
---

# Desafío 2: *Word Embeddings* para *Hamlet* de Shakespeare

[Archivo original en Colab](https://colab.research.google.com/drive/1-3nsIWYq2D5WzH5Ume3_fTlERh8xvQrz?usp=sharing#scrollTo=leSnYEBkCsii)

El objetivo principal de este desafío es explorar las relaciones semánticas dentro del texto de "Hamlet" aplicando *Word2Vec*, una popular técnica de *Word Embeddings*. El *notebook* guía a través de los siguientes pasos:

- **Extracción y preprocesamiento de texto**: aislamiento del diálogo de la versión del corpus *Gutenberg* de "Hamlet", limpieza del texto y aplicación de tokenización, lematización y eliminación de *stopwords*.
- **Entrenamiento del modelo *Word2Vec***: entrenamiento de un modelo *Word2Vec* *skip-gram* utilizando la librería *Gensim* para aprender representaciones vectoriales densas (*embeddings*) para palabras basándose en sus patrones de co-ocurrencia.
- **Análisis semántico**: evaluación de los *embeddings* aprendidos mediante consultas de similitud de palabras y pruebas de analogía.
- **Reducción de dimensionalidad y visualización**: uso de *t-SNE* para reducir los vectores de palabras de alta dimensión a 2D y 3D para la exploración visual de clústeres y relaciones de palabras.

---

## Objetivos del desafío

Este desafío fue desarrollado para cumplir con los siguientes ítems planteados:

- **Crear los propios vectores con *Gensim* basado en lo visto en clase con otro *dataset***: Se ha utilizado el corpus de "Hamlet" de *NLTK* para entrenar un modelo *Word2Vec* desde cero, generando *embeddings* vectoriales específicos para esta obra.
- **Probar términos de interés y explicar similitudes en el espacio de *embeddings***: Se han seleccionado un conjunto de palabras relevantes de "Hamlet" y se ha analizado su similitud a través de la función *most_similar* de *Gensim*, explicando las asociaciones contextuales y semánticas observadas. También se realizaron pruebas de analogía.
- **Graficarlos**: Se han generado visualizaciones en 2D y 3D de los *Word Embeddings* utilizando *t-SNE* y la librería *Plotly* para representar la proximidad semántica entre las palabras en un espacio reducido.
- **Obtener Conclusiones**: Se han derivado conclusiones detalladas a partir del preprocesamiento, el entrenamiento del modelo, los resultados de las pruebas de similitud/analogía y las visualizaciones de los *embeddings*.

---

## Características implementadas

- **Descarga automática de recursos *NLTK***: asegura que todos los corpus y modelos *NLTK* necesarios estén disponibles.
- **Preprocesamiento de texto robusto**: incluye limpieza basada en expresiones regulares para eliminar elementos estructurales (marcadores de acto/escena, etiquetas de hablantes, direcciones de escena) y pasos estándar de PLN (tokenización, minúsculas, lematización, eliminación de *stopwords*).
- **Entrenamiento de *Word2Vec* con seguimiento de pérdidas**: implementa una función de *callback* personalizada para monitorear la pérdida de entrenamiento por época, lo que proporciona información sobre la convergencia del modelo.
- **Consultas de similitud de palabras**: demuestra la funcionalidad *most_similar* para encontrar palabras semánticamente cercanas a términos de interés dentro del contexto de "Hamlet".
- **Aritmética vectorial para analogías**: intenta realizar analogías vectoriales (por ejemplo, "rey" - "hombre" + "mujer" = "reina") para probar la capacidad del modelo para capturar relaciones semánticas abstractas.
- **Visualizaciones interactivas *t-SNE***: genera diagramas de dispersión interactivos en 2D y 3D utilizando *Plotly*, lo que permite la exploración visual de clústeres de palabras en el espacio de *embedding*.

---

## Análisis y resultados

### Preprocesamiento del corpus

El *pipeline* de preprocesamiento extrajo aproximadamente 3735 oraciones efectivas de "Hamlet" para el entrenamiento de *Word2Vec*. Las observaciones clave incluyen:

- **Inclusión de elementos estructurales**: el preprocesamiento retuvo términos como *actus*, *primus*, *scoena*, *prima*, *enter*, *barnardo*, *francisco*, y abreviaturas de nombres de personajes (*fran*).
- **Ortografías arcaicas**: el modelo aprendió *embeddings* para ortografías arcaicas (ej., *vnfold*, *liue*), reflejando el vocabulario específico del inglés shakespeariano.
- **Limpieza efectiva**: la lematización y la eliminación de *stopwords* mejoraron significativamente la calidad de los *tokens*.

### Entrenamiento del modelo *Word2Vec*

El modelo *Word2Vec* fue entrenado con 3735 oraciones procesadas durante 100 épocas. La pérdida por época disminuyó progresivamente de ~274k a ~28k, indicando un aprendizaje efectivo y la convergencia del modelo. El tamaño del vocabulario final fue de 4145 palabras.

### Pruebas de similitud y analogía

#### Palabras más similares
Los resultados de *most_similar* demuestran la capacidad del modelo para capturar asociaciones semánticas muy específicas y contextualmente relevantes dentro de "Hamlet" para las palabras presentes en su vocabulario.

* Para *ophelia*, se observan asociaciones como *beautifed*, *idoll*, *orizons* y *nimph* (similitudes >0.73).
* La fuerte relación de *queen* con *willow* y *aslant* (ambas >0.77) vincula directamente a la descripción de la muerte de Ofelia narrada por Gertrudis.
* *ghost* muestra una conexión temática con *adulterate* (0.6547).
* Términos geopolíticos como *denmark* y *fortinbras* se asociaron con un léxico militar y político relevante, con similitudes a menudo superiores a 0.79.
* Se identificaron asociaciones menos intuitivas para *hamlet* (*vnbrac*, *doublet*), posiblemente ruido estadístico o la influencia de co-ocurrencias esporádicas.

#### Similitudes entre pares
La similitud coseno entre pares seleccionados cuantifica su cercanía en el espacio vectorial aprendido:

* La relación *king* - *claudius* (0.4284) indica una asociación semántica moderada.
* Pares como *hamlet* - *ophelia* (0.2943) y *king* - *queen* (0.2157) exhiben similitudes más débiles.
* La similitud *death* - *ghost* (0.2938) fue moderada-baja.

#### Limitaciones del vocabulario
Una limitación crítica observada es la ausencia de términos temáticos clave como *love*, *madness*, *revenge*, *skull* y *poison* en el vocabulario del modelo, probablemente debido a su frecuencia o al preprocesamiento.

#### Pruebas de analogía
Las pruebas de analogía (*king* - *man* + *woman* $\approx$ *queen* y *laertes* - *polonius* + *ghost* $\approx$ *hamlet*) no identificaron los términos canónicos esperados. Esto se atribuye principalmente a la especificidad del corpus de "Hamlet", que es relativamente pequeño y altamente especializado, limitando la capacidad del modelo para aprender subestructuras lineales generalizables para relaciones semánticas abstractas.

### Visualizaciones *t-SNE* de *Embeddings*

#### Gráfico *t-SNE* 2D (ver figura en el [archivo original en Colab](https://colab.research.google.com/drive/1-3nsIWYq2D5WzH5Ume3_fTlERh8xvQrz?usp=sharing#scrollTo=leSnYEBkCsii))
La proyección 2D muestra agrupaciones temáticas, con cierto solapamiento. Una agrupación densa de personajes prominentes (*hamlet*, *claudius*, *ghost*, *horatio*, *king*, *queen*, *gertrude*, *polonius*) resalta su prominencia en contextos comunes. Sin embargo, la bidimensionalidad impone limitaciones, con algunas palabras clave apareciendo más aisladas.
![Gráfico t-SNE 2D](Desafio_2/Figuras/t-SNE%202D.png)

#### Gráfico *t-SNE* 3D (ver figura en el [archivo original en Colab](https://colab.research.google.com/drive/1-3nsIWYq2D5WzH5Ume3_fTlERh8xvQrz?usp=sharing#scrollTo=leSnYEBkCsii))
Las visualizaciones tridimensionales ofrecen una representación más rica y matizada. Los clústeres observados en 2D se reafirman y organizan de manera más coherente. Se revelan subestructuras no evidentes en 2D: términos como *madness*, *death*, *revenge* y *father* parecen formar un subclúster temático. Palabras que parecían aisladas en 2D se integran de manera más natural.
![Gráfico t-SNE 3D](Desafio_2/Figuras/t-SNE%203D.png)

## Conclusiones
Este desafío ha demostrado la capacidad de los *Word Embeddings* generados con *Word2Vec* para capturar las relaciones semánticas dentro de un corpus literario específico como "Hamlet". Se cumplieron los objetivos del desafío al crear *embeddings* propios, probar términos y explicar similitudes, graficar los *embeddings* y obtener conclusiones. No obstante, a pesar de las limitaciones observadas en las pruebas de analogía (atribuibles a la especificidad del corpus), la riqueza de las relaciones contextuales capturadas justifica el enfoque de *Word Embeddings* para la exploración semántica de obras literarias específicas.

---
---

## Desafío 3 – Modelado de lenguaje a nivel de caracteres
---

### Consigna
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determinístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.

### Sugerencias
- Durante el entrenamiento, guiarse por el descenso de la perplejidad en los datos de validación para finalizar el entrenamiento. Para ello se provee un callback.
- Explorar utilizar SimpleRNN (celda de Elman), LSTM y GRU.
- *rmsprop* es el optimizador recomendado para la buena convergencia. No obstante se pueden explorar otros.

---

### Objetivo
Entrenar y comparar modelos de lenguaje *many-to-many* (**SimpleRNN**, **GRU** y **LSTM**) para predecir el siguiente carácter en una secuencia y generar nuevo texto, evaluando.

----

### Metodología propuesta

---
#### 1. Selección del corpus
Se seleccionó el **[ArXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset)** de Kaggle, un corpus textual representativo del dominio de investigación en inteligencia artificial, aprendizaje automático, informática y matemáticas.  
Para el entrenamiento, se compilaron en un único texto los 25 resúmenes más extensos de las cuatro categorías con más artículos.  
Este enfoque aseguró que el corpus fuera representativo y contuviera la variabilidad léxica y sintáctica necesaria para una buena generalización del modelo.

**Figura 1.** Top 15 categorías más frecuentes.  
![Top 15 categorías](./Desafio_3/figures/top15_categories_hist.png)

Como se muestra en la figura 1, las categorías dominantes en el corpus seleccionado son **Machine Learning**, **Computer Vision and Pattern Recognition**, **Computation and Language (Natural Language Processing)** y **Artificial Intelligence**. La figura 2 muestra que el corpus final tiene una distribución uniforme de la cantidad de palabras entre las cuatro categorías seleccionadas, lo que ayuda a evitar un sesgo significativo del modelo hacia una sola disciplina.

**Figura 2.** Cantidad de palabras por categoría.  
![Palabras por categoría](./Desafio_3/figures/top_categories_words_sum.png)

---

#### 2. Preprocesamiento del texto
El corpus fue normalizado y tokenizado carácter a carácter:

1. Conversión a minúsculas.  
2. Mapeo de cada carácter a un índice (`char2idx`) y su inverso (`idx2char`), guardados como archivos JSON.  
3. Definición de secuencias de contexto de **100 caracteres** (`max_context_size`).  
4. Generación de ejemplos de entrenamiento con ventana deslizante (*stride = 1*).  
5. División en:
   - **Entrenamiento:** 90% inicial.
   - **Validación:** 10% final, en bloques sin solapamiento.

---

#### 3. Diseño del modelo

##### 3.1. Modelos
Se implementaron tres variantes de redes recurrentes (`./Desafio_3/src/architectures.py`):

- **SimpleRNN:** entrada *one-hot*, capa recurrente `SimpleRNN` y capa `Dense`.  
- **GRU:** capa `Embedding`, dos capas `GRU` y salida `Dense`.  
- **LSTM:** capa `Embedding`, dos capas `LSTM` y salida `Dense`.

**Configuración común:**
- Optimizador: *RMSprop* (lr = 0.001).  
- Pérdida: *Categorical Crossentropy*.  
- Métrica adicional: *Perplejidad*.

##### 3.2. Callbacks
Se empleó un *callback* personalizado (`./Desafio_3/src/callbacks.py`) para:

- **Perplejidad:** calculada al final de cada época sobre validación:\
$$\mathrm{PPL}(X)=\exp\left(-\frac{1}{t}\sum_{i=1}^{t}\log p_{\theta}(w_i \mid w_{<i})\right)$$

- **Early Stopping:** con `patience = 3`.  
- **Guardado automático:** del mejor modelo en `models/`.

---

#### 4. Entrenamiento

**Figura 3.** Comparación de modelos durante el entrenamiento.  
![Comparación de modelos](./Desafio_3/figures/model_comparison.png)

- **SimpleRNN:** peor rendimiento, alta perplejidad y limitaciones en dependencias largas.  
- **GRU:** mejor rendimiento general, menor perplejidad en validación.  
- **LSTM:** buen rendimiento, pero con *overfitting* a partir de la época 10.

---

#### 5. Generación de texto
Se utilizó `./src/text_generator.py` para generar texto desde frases iniciales (*prompts*) como:

- `recurrent neural network`  
- `convolutional neural network`  
- `future researchs should`

**Estrategias**

##### Greedy Search (*temp = 0*)
Texto repetitivo y predecible.

**Ejemplo (GRU/LSTM):**
```
of the probability of the probability
```

**Ejemplo (SimpleRNN):**
```
to the the the the
```

##### Beam Search Estocástico
**Temp = 0.5:** más variedad pero aún con repeticiones.  
Ejemplo (GRU - future researchs should...):
```
future researchs should of the problem of the results in the problems of the problem and the computation...
```

**Temp = 1.5:** mayor creatividad; SimpleRNN incoherente, GRU y LSTM equilibrados.  
Ejemplo (GRU - future researchs should...):
```
future researchs should a related and dependent the clearning computer and the frameworks...
```

---

#### 6. Conclusiones
- **GRU** y **LSTM** superan claramente a **SimpleRNN** en la gestión de dependencias largas.  
- La mejor combinación fue **GRU + Beam Search Estocástico + Temp = 1.5**, logrando un balance entre coherencia y creatividad.  
- El modelado carácter a carácter presenta limitaciones para generar texto coherente en este dominio, pero es útil para evaluar el impacto de arquitectura y estrategia de decodificación.

---

## Desafío 4 - chatbot
### Modelo Seq2Seq (*Keras*) con un solo vocabulario y embedding compartido

Este proyecto consistió en entrenar un modelo encoder–decoder (*seq2seq*) basado en LSTM para generar respuestas en inglés a partir de pares de diálogo. La principal decisión de diseño fue utilizar un único tokenizador y un solo vocabulario tanto para las entradas como para las salidas, junto con una única capa de *embedding* compartida entre encoder y decoder. Esta estrategia simplificó el pipeline, evitó desalineaciones en los índices y redujo significativamente el consumo de memoria.

---

### Objetivo
* Desarrollar un sistema de diálogo sencillo en inglés, trabajando con pares pregunta → respuesta.
* Utilizar un único vocabulario (un solo Tokenizer) y una única matriz de embeddings (*GloVe* o *fastText*) reutilizada en ambas partes del modelo.
* Realizar la inferencia paso a paso, incorporando tokens especiales de inicio y fin de secuencia

---

### 1. Datos

* El dataset estuvo formado por conversaciones; de cada línea se extrajeron parejas consecutivas (entrada, salida).
* Se descartaron pares demasiado largos para evitar explotar memoria y estabilizar el entrenamiento (longitudes máximas típicas: 10–30 tokens).
* Se agregaron marcadores:
   * `<sos>` (start-of-sequence) al inicio de la salida para el decoder input.
   * `<eos>` (end-of-sequence) al final de la salida para el decoder target.

Limpieza recomendada
Se usaron minúsculas, normalización básica de contracciones en inglés, y filtrado de símbolos para dejar solo caracteres alfanuméricos/espacios. Fue importante reasignar los reemplazos (evitar funciones “no in-place”).

---

### 2. Vocabulario y tokenización

* Se usó un solo tokenizador entrenado con la unión de:

  * entradas,
  * salidas con `<eos>`,
  * salidas de entrada del decoder con `<sos>`.
* El tamaño del vocabulario se recortó a un máximo (p. ej., 8 000) y se reservó el índice 0 para padding.
* `<sos>` y `<eos>` debieron existir en el vocabulario y tener índices > 0.
* Todas las secuencias fueron paddeadas a longitudes fijas separadas: `max_input_len` para el encoder y `max_out_len` para el decoder.

---

### 3. Embeddings

* Se empleó un único conjunto de embeddings en inglés (*fastText*).
* Se construyó una sola matriz de tamaño *(vocab, dim)* usando el mismo diccionario del tokenizador.
* Detalle crítico: cuando se consultó el embedding de una palabra, se tuvo que tratar la palabra como unidad léxica, no como lista de caracteres. (En términos prácticos: la función que obtuvo embeddings tuvo que recibir un conjunto/lista de palabras, no un string suelto).
* Las palabras fuera del vocabulario de los embeddings quedaron con vector nulo; convino monitorear la cobertura (proporción de palabras con vector no nulo).

Elección del embedding
* fastText (p. ej., wiki-news-300d): ofreció mejor cobertura por subpalabras, a costa de mayor tamaño.
`class FasttextEmbeddings(WordsEmbeddings):
  WORD_TO_VEC_MODEL_TXT_PATH = 'cc.en.300.vec'
  PKL_PATH = 'fasttext.pkl'
  N_FEATURES = 300
  WORD_MAX_SIZE = 60`

---

### 4. Arquitectura del modelo

* Embedding compartido (no entrenable, con máscara de padding activada): una sola capa que transformó IDs de tokens en vectores, usada tanto por encoder como por decoder.
* Encoder: LSTM con 128 unidades (configurable), con `dropout` y `recurrent_dropout` típicamente en 0.2.
* Decoder: otra LSTM de 128 unidades que recibió el estado final del encoder. Produjo una secuencia de logits que se proyectó con una capa densa al tamaño del vocabulario compartido.
* Función de pérdida: entropía cruzada categórica sobre la salida del decoder (one-hot o soft labels).
* Métrica: `accuracy` a nivel de token (útil para seguimiento; no siempre correlacionó con calidad lingüística).
![Diagrama](./Desafio_4/images/model_plot.png)

---

### 5. Entrenamiento

* Se entrenó con *teacher forcing*: el decoder vio la secuencia de salida “real” desplazada por `<sos>`.
* Se realizó una partición 80:20
* Épocas típicas: 100
  
#### 5.1. Monitoreo
* Las curvas del accuracy y loss mostraron lo que pareció ser un overfitting, sin embargo, el accuracy no fue una buena métrica para lenguajes de procesamiento de lenguaje natural.
![Curvas](./Desafio_4/images/training_curves.png)

---

#### 5.2. Inferencia (decodificación)

* Se construyó un encoder de inferencia que, dado el input paddeado, devolvió los estados ocultos iniciales del decoder.
* El decoder de inferencia funcionó token a token:

  1. Se inició con `<sos>`.
  2. En cada paso se tomó el token previo, se lo pasó por la misma capa de Embedding compartida, se propagó en el LSTM junto con los estados, y se obtuvo una distribución sobre el vocabulario.
  3. Se seleccionó el siguiente token (*greedy*).
  4. Se detuvo al predecir `<eos>` o alcanzar la longitud máxima.

> Clave: el decoder de inferencia reutilizó exactamente las mismas capas y pesos del entrenamiento (Embedding, LSTM y Dense). No se crearon capas nuevas “en blanco”.

Ejemplos de inferencia en preguntas elaboradas por el usario (🧔🏽‍♂️) que responde el *chatbot*(🤖):

🧔🏽‍♂️ *"What do you do for a living?"*

🤖 *"i am a student"*

-
🧔🏽‍♂️ *"Do you read?"*

🤖 *"yes"*

-
🧔🏽‍♂️ *"Do you have any pet?"*

🤖 *"yes i have a tiger"*

-
🧔🏽‍♂️ *"Where are you from?"*
🤖 *"i am from the united states"*


---

### Conclusiones

Un solo vocabulario y un solo Embedding simplificaron el entrenamiento y la inferencia, evitando errores de índices y reduciendo la memoria. Con limpieza adecuada, cobertura de embeddings razonable y un pipeline de inferencia que reutilizó las mismas capas entrenadas, el sistema produjo respuestas muy coherentes para diálogos simples en inglés.
