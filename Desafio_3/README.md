# Desafíio 3
---
### Consigna
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.


### Sugerencias
- Durante el entrenamiento, guiarse por el descenso de la perplejidad en los datos de validación para finalizar el entrenamiento. Para ello se provee un callback.
- Explorar utilizar SimpleRNN (celda de Elman), LSTM y GRU.
- rmsprop es el optimizador recomendado para la buena convergencia. No obstante se pueden explorar otros.
---

### Consigna
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.


### Sugerencias
- Durante el entrenamiento, guiarse por el descenso de la perplejidad en los datos de validación para finalizar el entrenamiento. Para ello se provee un callback.
- Explorar utilizar SimpleRNN (celda de Elman), LSTM y GRU.
- rmsprop es el optimizador recomendado para la buena convergencia. No obstante se pueden explorar otros.



# Metodología propuesta
#### 1. Selección del corpus:
Se seleccionó un corpus textual que sirviera como base para el entrenamiento del modelo de lenguaje. Este corpus debía ser representativo del dominio de interés y contener suficiente variabilidad léxica y sintáctica para permitir una generalización adecuada del modelo.

#### 2. Preprocesamiento del texto:
El corpus fue sometido a un proceso de preprocesamiento que incluyó:
* Conversión a minúsculas.
* Tokenización carácter a carácter.
* Codificación de los caracteres mediante índices enteros.
* Estructuración del conjunto de datos en secuencias de longitud fija.
* División en conjuntos de entrenamiento y validación.

#### 3. Diseño del modelo:
Se exploraron distintas arquitecturas de redes neuronales recurrentes para la tarea de modelado de lenguaje. En particular, se evaluaron modelos basados en *SimpleRNN*, *LSTM* y *GRU*, integrando capas de proyección y activación, y utilizando codificación *one-hot* como representación de entrada.

#### 4. Entrenamiento:
Los modelos fueron entrenados aplicando la perplejidad desde un *Callback ad-hoc*.

#### 5. Generación de texto
Los modelos entrenados fueron utilizados para generar nuevas secuencias a partir de contextos semilla. Se implementaron distintas estrategias de decodificación:
* *Greedy search:* selección del carácter más probable en cada paso.
* *Beam search determinista:* exploración de múltiples trayectorias con selección sistemática de los mejores candidatos.
* *Beam search estocástico:* incorporación de muestreo probabilístico controlado mediante un parámetro de temperatura, permitiendo analizar su impacto en la diversidad y coherencia de las secuencias generadas.

#### 6. Conclusiones: 
Se genearon las principales conclusiones basada en los resultados encontrados. 


# Desafíio 3
---
## Modelo de lenguaje a nivel de caracteres – Comparación entre SimpleRNN, GRU y LSTM
Este proyecto desarrolla y compara modelos de lenguaje carácter a carácter usando redes neuronales recurrentes (*RNN*), como *SimpleRNN*, *GRU* y *LSTM*. Se entrena sobre un corpus de resúmenes de artículos científicos, midiendo métricas como accuracy, loss y perplejidad para evaluar el rendimiento, y generando texto con diferentes estrategias de decodificación para analizar la coherencia y diversidad del texto. El proyecto busca determinar la arquitectura y la estrategia de generación más adecuadas para esta tarea, concluyendo que el modelado a nivel de carácter es poco efectivo para generar texto coherente en este dominio.

---
## Objetivo
Entrenar modelos de lenguaje a nivel de carácter para predecir el siguiente carácter en una secuencia de texto y generar nuevas secuencias. Se evalúa cómo la arquitectura de la red y las estrategias de decodificación afectan la coherencia y diversidad del texto generado.

---
## Descripción y análisis del Corpus
Se utilizó un corpus de texto creado a partir de un subconjunto del "ArXiv Scientific Research Papers Dataset". Para ello, se seleccionaron aleatoriamente 500 artículos del conjunto de datos original. De estos, se tomaron los 25 resúmenes más extensos de cada una de las cuatro categorías dominantes para compilar un texto continuo.

## Exploración del corpus
**Figura 1.** Top 15 categorías más frecuentes en el dataset [cite: 4_modelo_lenguaje_char.ipynb].

![Top 15 categorías](./figures/top15_categories_hist.png)

Como se muestra en la figura 1, las categorías dominantes en el corpus seleccionado son Machine Learning, Computer Vision and Pattern Recognition, Computation and Language (Natural Language Processing) y Artificial Intelligence. La categoría Machine Learning (Statistics) fue recategorizada a Machine Learning para unificar los datos.

**Figura 2.** Cantidad de palabras según las categorías seleccionadas [cite: 4_modelo_lenguaje_char.ipynb].

![Palabras por categoría](./figures/top_categories_words_sum.png)

La figura 2 muestra que el corpus final tiene una distribución uniforme de la cantidad de palabras entre las cuatro categorías seleccionadas, lo que ayuda a evitar un sesgo significativo del modelo hacia una sola disciplina. El vocabulario resultante tiene un total de 68 caracteres únicos, y se observó la presencia de términos específicos como *xgboost*, que son representativos de las disciplinas del corpus.

---
## Metodología
# Desafío 3 – Modelado de Lenguaje a Nivel de Caracteres

## 1. Objetivo
Entrenar y comparar modelos de lenguaje carácter a carácter (**SimpleRNN**, **GRU** y **LSTM**) para predecir el siguiente carácter en una secuencia y generar nuevo texto.  
Se evalúa cómo la arquitectura y las estrategias de decodificación afectan la coherencia y diversidad del texto generado.

---

## 2. Selección del corpus
Se utilizó el dataset [**ArXiv Scientific Research Papers Dataset**](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset), compuesto por artículos de arXiv en áreas como inteligencia artificial, aprendizaje automático, informática y matemáticas.

Para este trabajo:
- Se seleccionaron **500 artículos**.
- De cada una de las **cuatro categorías dominantes** se tomaron los **25 resúmenes más extensos**.
- Los resúmenes fueron compilados en un único texto continuo para su análisis.

**Figura 1.** Top 15 categorías más frecuentes.  
![Top 15 categorías](./figures/top15_categories_hist.png)

**Figura 2.** Cantidad de palabras por categoría.  
![Palabras por categoría](./figures/top_categories_words_sum.png)

---

## 3. Preprocesamiento del texto
El corpus fue normalizado y tokenizado carácter a carácter:

1. Conversión a minúsculas.
2. Mapeo de cada carácter a un índice (`char2idx`) y su inverso (`idx2char`), guardados como archivos JSON.
3. Definición de secuencias de contexto de **100 caracteres** (`max_context_size`).
4. Generación de ejemplos de entrenamiento con ventana deslizante (*stride=1*).
5. División en:
   - **Entrenamiento:** 90% inicial.
   - **Validación:** 10% final, en bloques sin solapamiento.

---

## 4. Diseño de arquitecturas
Se implementaron tres variantes de redes recurrentes ([`src/architectures.py`](./src/architectures.py)):

- **SimpleRNN:** entrada *one-hot*, capa recurrente `SimpleRNN` y capa `Dense`.
- **GRU:** capa `Embedding`, dos capas `GRU` y salida `Dense`.
- **LSTM:** capa `Embedding`, dos capas `LSTM` y salida `Dense`.

**Configuración común:**
- Optimizador: *RMSprop* (lr = 0.001).
- Pérdida: *Categorical Crossentropy*.
- Métrica adicional: *Perplejidad*.

---

## 5. Entrenamiento y callbacks
Se empleó un *callback* personalizado ([`src/callbacks.py`](./src/callbacks.py)) para:

- **Perplejidad:** calculada al final de cada época sobre validación:

$$
\mathrm{PPL}(X) = \exp\!\left(-\frac{1}{t}\sum_{i=1}^{t}\log p_{\theta}(w_i \mid w_{<i})\right)
$$

- **Early Stopping:** con `patience=3`.
- **Guardado automático:** del mejor modelo en `models/`.

---

## 6. Resultados de entrenamiento

**Figura 3.** Comparación de modelos durante el entrenamiento.  
![Comparación de modelos](./figures/model_comparison.png)

- **SimpleRNN:** peor rendimiento, alta perplejidad y limitaciones en dependencias largas.
- **GRU:** mejor rendimiento general, menor perplejidad en validación.
- **LSTM:** buen rendimiento, pero con *overfitting* a partir de la época 10.

---

## 7. Generación de texto

Se utilizó [`src/text_generator.py`](./src/text_generator.py) para generar texto desde frases iniciales (*prompts*) como:

* `recurrent neural network`
* `convolutional neural network`
* `future researchs should`

**Estrategias:**

### 7.1 Greedy Search *(temp = 0)*
Texto repetitivo y predecible.

**Ejemplo (GRU/LSTM):** *of the probability of the probability...*
**Ejemplo (SimpleRNN):** *to the the the the...*


### 7.2 Beam Search Estocástico
**Temp = 0.5:** más variedad pero aún con repeticiones.  
Ejemplo (GRU): *future researchs should of the problem of the results in the problems of the problem and the computation...*
```

**Temp = 1.5:** mayor creatividad; SimpleRNN incoherente, GRU y LSTM equilibrados.  
Ejemplo (GRU): *future researchs should a related and dependent the clearning computer and the frameworks...*


---

## 8. Conclusiones
- **GRU** y **LSTM** superan claramente a **SimpleRNN** en la gestión de dependencias largas.
- La mejor combinación fue **GRU + Beam Search Estocástico + Temp = 1.5**, logrando un balance entre coherencia y creatividad.
- El modelado carácter a carácter presenta limitaciones para generar texto coherente en este dominio, pero es útil para evaluar el impacto de arquitectura y estrategia de decodificación.
