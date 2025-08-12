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
![Top 15 categorías](top15_categories_hist.png)

Como se muestra en la figura 1, las categorías dominantes en el corpus seleccionado son Machine Learning, Computer Vision and Pattern Recognition, Computation and Language (Natural Language Processing) y Artificial Intelligence. La categoría Machine Learning (Statistics) fue recategorizada a Machine Learning para unificar los datos.

**Figura 2.** Cantidad de palabras según las categorías seleccionadas [cite: 4_modelo_lenguaje_char.ipynb].
![Palabras por categoría](./figures/top_categories_words_sum.png)

La figura 2 muestra que el corpus final tiene una distribución uniforme de la cantidad de palabras entre las cuatro categorías seleccionadas, lo que ayuda a evitar un sesgo significativo del modelo hacia una sola disciplina. El vocabulario resultante tiene un total de 68 caracteres únicos, y se observó la presencia de términos específicos como *xgboost*, que son representativos de las disciplinas del corpus.

---
## Metodología
### Preprocesamiento y estructuración del texto
El corpus de texto se preprocesó y estructuró para el entrenamiento de los modelos.

**Tokenización:** Cada carácter del texto fue mapeado a un índice numérico único utilizando un diccionario llamado char2idx, y su inverso, idx2char, para la decodificación. Estos diccionarios se guardaron posteriormente en la carpeta models como archivos JSON (*char2idx.json* y *idx2char.json*).

**Secuenciación de datos:** Se definió un tamaño de contexto de 100 caracteres (*max_context_size*).

**Conjunto de entrenamiento:** Se usó una ventana deslizante con stride=1 para generar secuencias superpuestas de entrada y salida, permitiendo que la red aprenda a predecir el siguiente carácter en cada paso de la secuencia.

**Conjunto de validación:** Se reservó el 10% final del corpus para la validación, creando bloques de 100 caracteres sin superposición para una evaluación objetiva.

### Diseño de arquitecturas
Se implementaron y compararon tres arquitecturas de redes recurrentes, definidas en el script architectures.py dentro de la carpeta src.

**SimpleRNN:** Utiliza la codificación one-hot para la representación de los caracteres de entrada, seguida de una capa SimpleRNN y una capa Dense para la predicción de la salida.

**GRU y LSTM:** Estas arquitecturas, a diferencia de la *SimpleRNN*, utilizan una capa de Embedding para la representación de los caracteres, seguida de dos capas recurrentes (*GRU* o *LSTM*) y capas Dense para la salida.

El entrenamiento se configuró para usar el optimizador *RMSprop* con una tasa de aprendizaje de 0.001.

### Callback para entrenamiento##
Para monitorear y controlar el entrenamiento, se usó un callback personalizado llamado PplCallback (definido en callbacks.py), el cual implementa:

Perplejidad: A diferencia de la métrica loss, se calculó la perplejidad al final de cada época sobre el conjunto de validación para una medición más precisa del rendimiento del modelo de lenguaje. La perplejidad se calcula con la fórmula:

$$$$
\\mathrm{PPL}(X) = \\exp\\left( -\\frac{1}{t} \\sum\_{i=1}^{t} \\log p\_{\\theta}\\left( w\_i \\middle| w\_{\<i} \\right) \\right)
$$$$

**Early stopping:** El entrenamiento se detiene si la perplejidad en el conjunto de validación no mejora durante un número predefinido de épocas (patience=3).

**Guardado del modelo:** El modelo con la mejor perplejidad en validación se guarda automáticamente en la carpeta models.

---
## Resultados
### Rendimiento de las arquitecturas
La figura 3 muestra la comparación de las métricas de rendimiento durante el entrenamiento.

**Figura 3.** Estadísticas de los modelos en función de las épocas de entrenamiento [cite: model_comparison.png].
![Comparación de modelos](figures/model_comparison.png)

*SimpleRNN* mostró el peor desempeño, estabilizándose con valores de loss y perplejidad significativamente más altos, lo que confirma su limitación para manejar dependencias de largo plazo.

*GRU* presentó el mejor desempeño, alcanzando la menor perplejidad en validación, sugiriendo una mejor capacidad para generalizar con este corpus.

*LSTM* también tuvo un buen rendimiento, pero mostró indicios de overfitting a partir de la época 10, donde su perplejidad en validación comenzó a aumentar mientras que el accuracy de entrenamiento seguía mejorando.


### Ejemplos de generación de texto
Se utilizó el script text_generator.py para generar texto a partir de las frases iniciales *recurrent neural network*, *convolutional neural network* y *future researchs should*.

Generación por Greedy Search (temperatura=0)
Esta estrategia, que selecciona el carácter con mayor probabilidad en cada paso, resultó en secuencias altamente repetitivas y predecibles, como "of the probability of the probability" en los modelos GRU y LSTM, y "to the the the the" en el modelo SimpleRNN [cite: 4_modelo_lenguaje_char.ipynb].

Generación por Beam Search estocástico
**Temperatura baja (TEMP=0.5):** La calidad del texto mejoró, mostrando más variedad de palabras, aunque aún con repeticiones. Por ejemplo, el modelo *GRU* generó la secuencia "future researchs should of the problem of the results in the problems of the problem and the computation of the computation" [cite: 4_modelo_lenguaje_char.ipynb].

**Temperatura alta (TEMP=1.5):** La *SimpleRNN* generó texto caótico e incoherente, con palabras inexistentes. En cambio, los modelos *GRU* y *LSTM* mostraron una gran creatividad y coherencia. Por ejemplo, el modelo *GRU* generó: "future researchs should a related and dependent the clearning computer and the frameworks where the distrated to the propos" [cite: 4_modelo_lenguaje_char.ipynb].

---
## Conclusiones
La elección de la arquitectura es crucial, siendo GRU y LSTM las más adecuadas para modelar este tipo de texto, superando a SimpleRNN en la gestión de dependencias largas [cite: 4_modelo_lenguaje_char.ipynb]. El modelo GRU fue el más eficiente y el de mejor rendimiento.

La estrategia de decodificación más eficaz fue el Beam Search estocástico con una temperatura alta, ya que logró un equilibrio óptimo entre la coherencia y la creatividad del texto generado. Esto subraya la importancia de combinar una arquitectura robusta con una estrategia de decodificación adecuada para obtener resultados de alta calidad [cite: 4_modelo_lenguaje_char.ipynb].
