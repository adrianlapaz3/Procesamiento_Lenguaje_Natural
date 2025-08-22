Modelo Seq2Seq (*Keras*) con **un solo vocabulario** y **Embedding compartido**

Este proyecto entrena un modelo encoder–decoder (*seq2seq*) con LSTM para responder en **inglés** a partir de pares de diálogo. La decisión de diseño central es usar **un único tokenizador y un único diccionario** para entradas y salidas, y **una única capa de Embedding** compartida por encoder y decoder. Esto simplifica el pipeline, evita desalineaciones de índices y reduce memoria.

---

## Objetivo

* Construir un sistema de diálogo simple en inglés basado en pares *pregunta → respuesta*.
* Mantener un **único vocabulario** (un solo `Tokenizer`) y una **única matriz de embeddings** (GloVe o fastText) reutilizada en ambas mitades del modelo.
* Realizar inferencia paso a paso con tokens especiales de comienzo y fin de secuencia.

---

## Datos

* El dataset está formado por conversaciones; de cada línea se extraen parejas consecutivas **(entrada, salida)**.
* Se descartan pares demasiado largos para evitar explotar memoria y estabilizar el entrenamiento (longitudes máximas típicas: 10–30 tokens).
* Se agregan marcadores:

  * `<sos>` (start-of-sequence) **al inicio** de la salida *para el decoder input*.
  * `<eos>` (end-of-sequence) **al final** de la salida *para el decoder target*.

**Limpieza recomendada**
Minúsculas, normalización básica de contracciones en inglés, y filtrado de símbolos para dejar solo caracteres alfanuméricos/espacios. Es importante **reasignar** los reemplazos (evitar funciones “no in-place”).

---

## Vocabulario y tokenización

* Se usa **un solo tokenizador** entrenado con la unión de:

  * entradas,
  * salidas con `<eos>`,
  * salidas de entrada del decoder con `<sos>`.
* El tamaño del vocabulario se recorta a un máximo (p. ej., 8 000) y se **reserva el índice 0** para padding.
* `<sos>` y `<eos>` deben existir en el vocabulario y tener índices **> 0**.
* Todas las secuencias se **padean** a longitudes fijas separadas: `max_input_len` para el encoder y `max_out_len` para el decoder.

---

## Embeddings

* Se emplea **un único conjunto de embeddings en inglés** (*fastText*).
* Se construye **una sola matriz** de tamaño *(vocab, dim)* usando el **mismo diccionario** del tokenizador.
* Detalle crítico: cuando se consulta el embedding de una palabra, hay que tratar la palabra como **unidad léxica**, no como lista de caracteres. (En términos prácticos: la función que obtiene embeddings debe recibir un **conjunto/lista de palabras**, no un string suelto).
* Las palabras fuera del vocabulario de los embeddings quedan con vector nulo; conviene monitorear la **cobertura** (proporción de palabras con vector no nulo).

**Elección del embedding**
* **fastText** (p. ej., wiki-news-300d): mejor cobertura por subpalabras, a costa de mayor tamaño.
`class FasttextEmbeddings(WordsEmbeddings):
  WORD_TO_VEC_MODEL_TXT_PATH = 'cc.en.300.vec'
  PKL_PATH = 'fasttext.pkl'
  N_FEATURES = 300
  WORD_MAX_SIZE = 60`

---

## Arquitectura del modelo

* **Embedding compartido** (no entrenable, con máscara de padding activada): una sola capa que transforma IDs de tokens en vectores, usada tanto por encoder como por decoder.
* **Encoder**: LSTM con 128 unidades (configurable), con `dropout` y `recurrent_dropout` típicamente en 0.2.
* **Decoder**: otra LSTM de 128 unidades que recibe el estado final del encoder. Produce una **secuencia de logits** que se proyecta con una capa densa al tamaño del vocabulario compartido.
* **Función de pérdida**: entropía cruzada categórica sobre la salida del decoder (one-hot o soft labels).
* **Métrica**: `accuracy` a nivel de token (útil para seguimiento; no siempre correlaciona con calidad lingüística).

---

## Entrenamiento

* Entrenar con *teacher forcing*: el decoder ve la secuencia de salida “real” desplazada por `<sos>`.
* Partición 80:20 
* Épocas típicas: 100
* 
**Monitoreo**
* Las gracias del **accuracy** y **loss** muestran lo que parece ser un overfitting, sin embargo el accuracy no es una buena metrica para lenguajes de procesamiento de lenguaje natural.

---

## Inferencia (decodificación)

* Se construye un **encoder de inferencia** que, dado el input paddeado, devuelve los estados ocultos iniciales del decoder.
* El **decoder de inferencia** funciona **token a token**:

  1. Se inicia con `<sos>`.
  2. En cada paso se toma el token previo, se lo pasa por la misma capa de Embedding compartida, se propaga en el LSTM junto con los estados, y se obtiene una distribución sobre el vocabulario.
  3. Se selecciona el siguiente token (greedy o con estrategia top-k/temperatura).
  4. Se detiene al predecir `<eos>` o alcanzar la longitud máxima.

> Clave: el decoder de inferencia **reutiliza exactamente** las **mismas capas y pesos** del entrenamiento (Embedding, LSTM y Dense). No se crean capas nuevas “en blanco”.

---

## Directorios sugeridos

* `data/` o raíz del proyecto: dataset JSON.
* `embeddings/`: archivos .txt/.vec oficiales (y, si se desea, su cache local).
* `images/`: figuras (diagrama del modelo, curvas de entrenamiento).
* Notebook o script principal.

---

## Problemas comunes y cómo resolverlos

1. **El modelo “copia” la entrada o responde siempre igual**

   * Matriz de embeddings mal construida (vectores nulos por pasar cadenas en lugar de listas de palabras).
   * Cobertura baja del vocabulario en los embeddings.
   * Dataset con muchos pares *entrada=salida*.
   * Soluciones: corregir la consulta de embeddings, revisar cobertura, limpiar pares idénticos, usar dropout, y (si procede) añadir muestreo top-k en inferencia.

2. **Error al cargar un `.pkl` de embeddings (archivo HTML)**

   * Sucede cuando se baja un “pickle” desde enlaces que devuelven páginas intermedias.
   * Solución: usar los **.txt/.vec oficiales** y convertir localmente; si se detecta corrupción, regenerar.

3. **Desfase de dimensiones en `Embedding`**

   * La matriz de embeddings no coincide con `input_dim`.
   * Solución: asegurar que **`num_words`** y el **tamaño de la primera dimensión** de la matriz sean exactamente iguales; que todo provenga del **mismo tokenizador**.

4. **Índices de tokens especiales**

   * `<sos>` y `<eos>` deben existir en el vocabulario y no ser 0 (el 0 se usa para padding con máscara).

5. **Errores de forma en entradas**

   * Las formas de entrada deben ser tuplas (p. ej., “(max\_len,)”) y el tipo entero.
   * La máscara en Embedding debe ignorar el padding (`mask_zero=True`).

---

## Limitaciones y mejoras

* El modelo seq2seq con LSTM puede quedarse corto en cobertura semántica profunda o respuestas largas.
* Posibles mejoras: atención (Bahdanau/Luong), *beam search*, regularización extra, fine-tuning de la capa de Embedding (parcial), o migración a arquitecturas Transformer.

---

## Resumen

* Un **solo vocabulario** y **un solo Embedding** simplifican el entrenamiento y la inferencia, evitando errores de índices y reduciendo memoria.
* La **consistencia** entre tokenizador, matriz de embeddings y capas (mismos tamaños e índices) es la clave para resultados estables.
* Con limpieza adecuada, cobertura de embeddings razonable y un pipeline de inferencia que reutilice las mismas capas entrenadas, el sistema produce respuestas coherentes para diálogos simples en inglés.

