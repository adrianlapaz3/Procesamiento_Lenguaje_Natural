## ADRIÁN LAPAZ (1706)
# Desafío 1 

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
