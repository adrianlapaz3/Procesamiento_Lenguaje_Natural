# Desafío 1 
# ADRIÁN LAPAZ (1706)

Este proyecto aborda un desafío de Procesamiento del Lenguaje Natural (PLN) utilizando el conjunto de datos **20 Newsgroups**. El objetivo es explorar técnicas de vectorización, clasificación de texto y análisis de similaridad semántica entre palabras.

## Descripción del desafío

El desafío se divide en tres partes principales:
1.  **Similaridad de documentos**: vectorizar el *corpus* de texto y analizar la similaridad del coseno entre documentos para evaluar la coherencia temática.
2.  **Clasificación de texto**: entrenar y optimizar modelos de clasificación Naïve Bayes (*MultinomialNB* y *ComplementNB*) para predecir la categoría de un documento, maximizando la métrica *f1-score macro*.
3.  **Similaridad de Palabras**: transponer la matriz documento-término para crear vectores de palabras y analizar las relaciones semánticas entre términos seleccionados manualmente.


## Resultados principales

### 1. Similaridad entre documentos
Se seleccionaron 5 documentos al azar y se calcularon sus 5 vecinos más similares usando la similaridad del coseno.

- **Coherencia temática**: se observó que los documentos con mayor similaridad a menudo pertenecen a la misma categoría (*rec.sport.hockey*, *rec.sport.baseball*). Por ejemplo, un documento de la categoría *rec.sport.hockey* tuvo sus 5 documentos más similares también pertenecientes a esa categoría, con valores de coseno de hasta **0.37**.
- **Bajos valores de coseno**: En general, los valores de similaridad fueron moderados o bajos (entre 0.14 y 0.37). Esto sugiere que, si bien la similaridad del coseno puede agrupar temas, podría no ser el método más robusto para una clasificación precisa por sí solo.

### 2. Clasificación con Naïve Bayes
Se utilizó **optimización bayesiana (*BayesSearchCV*)** para encontrar los mejores hiperparámetros tanto para el vectorizador *TfidfVectorizer* como para los clasificadores *MultinomialNB* y *ComplementNB*.

- **Rendimiento en entrenamiento (CV, Cross-Validation)**:
  - **MultinomialNB**: mejor *F1-score* (CV) de **0.7626**.
  - **ComplementNB**: mejor *F1-score* (CV) de **0.7661**.
- **Rendimiento en los datos de testeo**:
  - **MultinomialNB**: *F1-score* en Test de **0.6876**.
  - **ComplementNB**: *F1-score* en Test de **0.6969**.

Ambos modelos mostraron un rendimiento muy similar, aunque *ComplementNB* fue marginalmente superior. El análisis de hiperparámetros reveló que *ComplementNB* logró su mejor rendimiento con un filtrado dinámico de vocabulario y un suavizado mayor, sugiriendo una mayor robustez frente al ruido léxico.

### 3. Similaridad entre palabras
Se analizó la similaridad entre 5 palabras seleccionadas (*ball*, *doctor*, *python*, *space*, *water*) tras transponer la matriz *TF-IDF*.

- **Captura de contexto temático y/o semántica**: el análisis demostró la capacidad del modelo para identificar relaciones contextuales muy específicas.
  - La asociación más fuerte fue entre **python** y **monty** (similaridad de **0.7138**), una clara referencia semántica al grupo de comedia *"Monty Python"*.
  - Se encontraron fuertes agrupaciones temáticas, como **doctor** con **receptionist** (0.4392) y **space** con **nasa** (0.3304).
  - La palabra **water** se asoció con términos de infraestructura urbana como **towers**, **dpw** y **croton**, mostrando cómo el modelo captura el contexto específico del *corpus*.

---

## Metodología y herramientas

- **Librerías principales**: *scikit-learn*, *numpy*, *skopt*.
- **Vectorización**: *TfidfVectorizer*.
- **Modelos**: *MultinomialNB*, *ComplementNB*.
- **Métrica de evaluación**: *f1_score* (*macro average*).
- **Técnica de optimización**: búsqueda bayesiana (*BayesSearchCV*) para una sintonización eficiente de hiperparámetros.

## Conclusión
La **similaridad del coseno** es útil para la exploración temática, pero los modelos de clasificación como **Naïve Bayes** son superiores para tareas de predicción. Además, el análisis de similaridad de palabras sobre la matriz transpuesta reveló ser una técnica muy poderosa para descubrir **relaciones semánticas** en el texto.
