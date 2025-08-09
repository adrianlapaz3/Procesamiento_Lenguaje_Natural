# architectures.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, SimpleRNN, Dropout, Dense,
    TimeDistributed, CategoryEncoding
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from typing import Optional


class BaseRNNModel:
    def __init__(self,
                 vocab_size: int,
                 sequence_length: int,
                 embedding_dim: int = 64,
                 rnn_units: int = 64,
                 dropout_rate: float = 0.2,
                 projection_dim: Optional[int] = None,
                 activation: str = 'relu',
                 optimizer_name: str = 'rmsprop',
                 learning_rate: float = 0.001):

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.projection_dim = projection_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name.lower()
        self.model = None

    def get_optimizer(self):
        if self.optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Optimizador '{self.optimizer_name}' no soportado.")

    def compile_model(self):
        self.model.compile(
            loss=SparseCategoricalCrossentropy(),
            optimizer=self.get_optimizer(),
            metrics=['accuracy']
        )

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Modelo no ha sido construido.")


class GRUModel(BaseRNNModel):
    def build(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size + 1,
                            output_dim=self.embedding_dim,
                            input_shape=(self.sequence_length,)))
        model.add(GRU(self.rnn_units, return_sequences=True,
                      dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
        model.add(GRU(self.rnn_units, return_sequences=True,
                      dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))

        if self.projection_dim:
            model.add(Dense(self.projection_dim, activation=self.activation))

        model.add(Dense(self.vocab_size + 1, activation='softmax'))
        self.model = model
        self.compile_model()


class LSTMModel(BaseRNNModel):
    def build(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size + 1,
                            output_dim=self.embedding_dim,
                            input_shape=(self.sequence_length,)))
        model.add(LSTM(self.rnn_units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.rnn_units, return_sequences=True))

        if self.projection_dim:
            model.add(Dense(self.projection_dim, activation=self.activation))

        model.add(Dense(self.vocab_size + 1, activation='softmax'))
        self.model = model
        self.compile_model()


class SimpleRNNModel(BaseRNNModel):
    def build(self):
        model = Sequential()
        model.add(TimeDistributed(CategoryEncoding(num_tokens=self.vocab_size,
                                                   output_mode='one_hot'),
                                  input_shape=(self.sequence_length, 1)))
        model.add(SimpleRNN(self.rnn_units, return_sequences=True,
                            dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))

        if self.projection_dim:
            model.add(Dense(self.projection_dim, activation=self.activation))

        model.add(Dense(self.vocab_size, activation='softmax'))
        self.model = model
        self.compile_model()

