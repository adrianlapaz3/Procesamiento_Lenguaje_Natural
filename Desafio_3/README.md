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

# **Desafío 3**

## **Metodología propuesta**

### 1. Selección del corpus
Se utilizó el **ArXiv Scientific Research Papers Dataset** de Kaggle. El corpus se construyó a partir de los 25 resúmenes más extensos de las cuatro categorías con más artículos:
* **Machine Learning**
* **Computer Vision and Pattern Recognition**
* **Computation and Language (Natural Language Processing)**
* **Artificial Intelligence**

Se unificó la categoría **Machine Learning (Statistics)** en **Machine Learning**. La distribución de palabras por categoría es uniforme, lo que ayuda a evitar sesgos en el modelo.

---

### 2. Preprocesamiento del texto
El texto fue preprocesado para ser apto para el entrenamiento del modelo a nivel de carácter.

* **Construcción del vocabulario**: Se extrajeron todos los caracteres únicos del corpus y se convirtieron a minúsculas. Se crearon dos diccionarios: **`char2idx`** para mapear caracteres a índices numéricos y **`idx2char`** para la conversión inversa. El vocabulario resultante contiene 68 caracteres únicos.
* **Tokenización**: El corpus se convirtió en una secuencia de índices numéricos.
* **División del dataset y creación de secuencias**: Se usó el 90% del corpus para entrenamiento y el 10% restante para validación. La longitud del contexto se fijó en 100 caracteres.
    * **Entrenamiento**: Se utilizó una ventana deslizante con un `stride` de 1 para crear secuencias superpuestas, lo que permitió al modelo aprender de múltiples contextos similares.
    * **Validación**: Se crearon bloques consecutivos de 100 caracteres sin superposición para una evaluación objetiva.
    * **Estructura de datos**: Se implementó un esquema `many-to-many`, donde cada secuencia de entrada ($X$) se emparejó con una secuencia de salida ($y$) que consistía en la misma secuencia de entrada desplazada un carácter hacia adelante.

---

### 3. Diseño del modelo
Se evaluaron tres arquitecturas de redes neuronales recurrentes clásicas.

* **SimpleRNN**: Utiliza la codificación `one-hot` para la representación de los caracteres de entrada, seguida de una capa `SimpleRNN` y una capa `Dense` para la predicción de la salida.
* **GRU y LSTM**: Estas arquitecturas, a diferencia de la `SimpleRNN`, usan una capa `Embedding` para la representación de caracteres, seguida de dos capas recurrentes (`GRU` o `LSTM`) y capas `Dense` para la salida.

---

### 4. Entrenamiento
El entrenamiento se guió por el descenso de la perplejidad. Para ello, se desarrolló un **`Callback ad-hoc`** que implementó:

* **Perplejidad**: A diferencia de la pérdida de entropía cruzada, la perplejidad ofrece una medida más intuitiva del rendimiento del modelo de lenguaje, calculándose con la siguiente fórmula:
    $$
    \mathrm{PPL}(X) = \exp\left( -\\frac{1}{t} \\sum_{i=1}^{t} \\log p_{\\theta}\\left( w_i \\,\\middle|\\, w_{<i} \\right) \\right)
    $$
    Este enfoque proporcionó un criterio robusto para evaluar la capacidad de generalización del modelo.
* **Early Stopping**: El entrenamiento se detuvo si la perplejidad en el conjunto de validación no mejoraba después de un número predefinido de épocas (`patience=3`), lo que evitó el sobreajuste.
* **Persistencia del modelo**: El modelo con la menor perplejidad de validación se guardó automáticamente.

---

### 5. Generación de texto
Se utilizaron los modelos entrenados para generar nuevas secuencias a partir de tres textos semilla:
* "recurrent neural network"
* "convolutional neural network"
* "future researchs should"

Se aplicaron distintas estrategias de decodificación.

* **Greedy Search (determinista)**: Esta estrategia, que selecciona el carácter con mayor probabilidad en cada paso (`temperatura=0`), resultó en secuencias altamente repetitivas y predecibles, como "of the probability of the probability" en los modelos GRU y LSTM, y "to the the the the" en el modelo SimpleRNN.
* **Beam Search (estocástico)**: Se incorporó muestreo probabilístico controlado por la temperatura.
    * **Temperatura baja (`TEMP=0.5`)**: La calidad del texto mejoró con más variedad, aunque se mantuvieron algunas repeticiones. Por ejemplo, el modelo `GRU` generó la secuencia "future researchs should of the problem of the results in the problems of the problem and the computation of the computation".
    * **Temperatura alta (`TEMP=1.5`)**: El texto de la `SimpleRNN` fue caótico e incoherente, con palabras inexistentes. En cambio, los modelos `GRU` y `LSTM` mostraron un equilibrio entre coherencia y creatividad. Por ejemplo, el modelo `GRU` generó: "future researchs should a related and dependent the clearning computer and the frameworks where the distrated to the propos".

---

### 6. Conclusiones

La elección de la arquitectura del modelo es crucial. Los modelos `GRU` y `LSTM` superaron a `SimpleRNN` en rendimiento y capacidad para gestionar dependencias a largo plazo. El modelo `GRU` fue el más eficaz con este corpus, alcanzando la menor perplejidad. El `LSTM`, aunque tuvo un buen desempeño, mostró señales de sobreajuste.

La estrategia de decodificación más efectiva fue el **Beam Search estocástico con una temperatura alta**, ya que logró un equilibrio óptimo entre la coherencia y la creatividad en el texto generado. Esto subraya la importancia de combinar una arquitectura robusta con una estrategia de decodificación adecuada para obtener resultados de alta calidad.
