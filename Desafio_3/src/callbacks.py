import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PplCallback(keras.callbacks.Callback):
    '''
    Callback personalizado para calcular la perplejidad al final de cada época
    utilizando un conjunto de datos de validación.

    Además de evaluar el modelo, implementa detención temprana (early stopping)
    si la perplejidad no mejora tras un número determinado de épocas (definido por `patience`).
    También guarda automáticamente la versión del modelo con mejor desempeño.
    '''

    def __init__(self, val_data, history_ppl, maxlen, model_name='my_model', patience=3):
        self.val_data = val_data
        self.model_name = model_name
        self.history_ppl = history_ppl
        self.maxlen = maxlen

        self.target = []
        self.padded = []
        self.info = []
        self.min_score = np.inf
        self.patience_counter = 0
        self.patience = patience
        
        count = 0

        # Generación de subsecuencias y vectores objetivo (target)
        for seq in self.val_data:
            len_seq = len(seq)
            subseq = [seq[:i] for i in range(1, len_seq)]
            self.target.extend([seq[i] for i in range(1, len_seq)])

            if len(subseq) != 0:
                self.padded.append(pad_sequences(subseq, maxlen, padding='pre'))
                self.info.append((count, count + len_seq))
                count += len_seq

        self.padded = np.vstack(self.padded)

        # Información sobre el conjunto de validación preprocesado
        print(f"[INFO] Forma del conjunto validación: {self.padded.shape}")
        print(f"[INFO] Total de subsecuencias procesadas: {len(self.padded)}")
        print(f"[INFO] Longitud máxima de secuencia: {self.padded.shape[1]}")

    def on_epoch_end(self, epoch, logs=None):
        scores = []
        predictions = self.model.predict(self.padded, verbose=0)

        for start, end in self.info:
            probs = [predictions[idx_seq, -1, idx_vocab]
                     for idx_seq, idx_vocab in zip(range(start, end), self.target[start:end])]
            scores.append(np.exp(-np.sum(np.log(probs)) / (end - start)))

        current_score = np.mean(scores)
        self.history_ppl.append(current_score)
        print(f"[EPOCH {epoch+1}] Perplejidad media en validación: {current_score:.4f}")

        if current_score < self.min_score:
            self.min_score = current_score
            self.model.save(self.model_name + '.keras')
            print(f"[GUARDADO] Nuevo modelo con mejor perplejidad ({current_score:.4f}) almacenado como '{self.model_name}.keras'")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"[INFO] No se observó mejora. Paciencia: {self.patience_counter}/{self.patience}")

            if self.patience_counter == self.patience:
                print(f"[DETENCIÓN ANTICIPADA] Se detiene el entrenamiento tras {self.patience} épocas sin mejora.")
                self.model.stop_training = True
