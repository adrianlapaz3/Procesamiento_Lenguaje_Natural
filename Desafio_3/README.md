## ADRIÁN LAPAZ (1706)
---
# Desafío 3 – Modelado de lenguaje a nivel de caracteres
---

## Consigna
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determinístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.

## Sugerencias
- Durante el entrenamiento, guiarse por el descenso de la perplejidad en los datos de validación para finalizar el entrenamiento. Para ello se provee un callback.
- Explorar utilizar SimpleRNN (celda de Elman), LSTM y GRU.
- *rmsprop* es el optimizador recomendado para la buena convergencia. No obstante se pueden explorar otros.

---

## Objetivo
Entrenar y comparar modelos de lenguaje *many-to-many* (**SimpleRNN**, **GRU** y **LSTM**) para predecir el siguiente carácter en una secuencia y generar nuevo texto, evaluando.

----

## Metodología propuesta

---
### 1. Selección del corpus
Se seleccionó el **[ArXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset)** de Kaggle, un corpus textual representativo del dominio de investigación en inteligencia artificial, aprendizaje automático, informática y matemáticas.  
Para el entrenamiento, se compilaron en un único texto los 25 resúmenes más extensos de las cuatro categorías con más artículos.  
Este enfoque aseguró que el corpus fuera representativo y contuviera la variabilidad léxica y sintáctica necesaria para una buena generalización del modelo.

**Figura 1.** Top 15 categorías más frecuentes.  
![Top 15 categorías](./figures/top15_categories_hist.png)

Como se muestra en la figura 1, las categorías dominantes en el corpus seleccionado son **Machine Learning**, **Computer Vision and Pattern Recognition**, **Computation and Language (Natural Language Processing)** y **Artificial Intelligence**. La figura 2 muestra que el corpus final tiene una distribución uniforme de la cantidad de palabras entre las cuatro categorías seleccionadas, lo que ayuda a evitar un sesgo significativo del modelo hacia una sola disciplina.

**Figura 2.** Cantidad de palabras por categoría.  
![Palabras por categoría](./figures/top_categories_words_sum.png)

---

### 2. Preprocesamiento del texto
El corpus fue normalizado y tokenizado carácter a carácter:

1. Conversión a minúsculas.  
2. Mapeo de cada carácter a un índice (`char2idx`) y su inverso (`idx2char`), guardados como archivos JSON.  
3. Definición de secuencias de contexto de **100 caracteres** (`max_context_size`).  
4. Generación de ejemplos de entrenamiento con ventana deslizante (*stride = 1*).  
5. División en:
   - **Entrenamiento:** 90% inicial.
   - **Validación:** 10% final, en bloques sin solapamiento.

---

### 3. Diseño del modelo

#### 3.1. Modelos
Se implementaron tres variantes de redes recurrentes (`./src/architectures.py`):

- **SimpleRNN:** entrada *one-hot*, capa recurrente `SimpleRNN` y capa `Dense`.  
- **GRU:** capa `Embedding`, dos capas `GRU` y salida `Dense`.  
- **LSTM:** capa `Embedding`, dos capas `LSTM` y salida `Dense`.

**Configuración común:**
- Optimizador: *RMSprop* (lr = 0.001).  
- Pérdida: *Categorical Crossentropy*.  
- Métrica adicional: *Perplejidad*.

#### 3.2. Callbacks
Se empleó un *callback* personalizado (`./src/callbacks.py`) para:

- **Perplejidad:** calculada al final de cada época sobre validación:\
$$\mathrm{PPL}(X)=\exp\left(-\frac{1}{t}\sum_{i=1}^{t}\log p_{\theta}(w_i \mid w_{<i})\right)$$

- **Early Stopping:** con `patience = 3`.  
- **Guardado automático:** del mejor modelo en `models/`.

---

### 4. Entrenamiento

**Figura 3.** Comparación de modelos durante el entrenamiento.  
![Comparación de modelos](./figures/model_comparison.png)

- **SimpleRNN:** peor rendimiento, alta perplejidad y limitaciones en dependencias largas.  
- **GRU:** mejor rendimiento general, menor perplejidad en validación.  
- **LSTM:** buen rendimiento, pero con *overfitting* a partir de la época 10.

---

### 5. Generación de texto
Se utilizó `./src/text_generator.py` para generar texto desde frases iniciales (*prompts*) como:

- `recurrent neural network`  
- `convolutional neural network`  
- `future researchs should`

**Estrategias**

#### Greedy Search (*temp = 0*)
Texto repetitivo y predecible.

**Ejemplo (GRU/LSTM):**
```
of the probability of the probability
```

**Ejemplo (SimpleRNN):**
```
to the the the the
```

#### Beam Search Estocástico
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

### 6. Conclusiones
- **GRU** y **LSTM** superan claramente a **SimpleRNN** en la gestión de dependencias largas.  
- La mejor combinación fue **GRU + Beam Search Estocástico + Temp = 1.5**, logrando un balance entre coherencia y creatividad.  
- El modelado carácter a carácter presenta limitaciones para generar texto coherente en este dominio, pero es útil para evaluar el impacto de arquitectura y estrategia de decodificación.

---

