#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creating neural network models.

Author:
    Erik Johannes Husom

Date:
    2022-01-09

"""
import numpy as np
import tensorflow as tf
from keras_tuner import HyperModel
from tensorflow.keras import layers, models, optimizers
from tensorflow.random import set_seed

def cnn(
    input_x,
    input_y,
    output_length=1,
    kernel_size=2,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a CNN model architecture using Keras.

    Args:
        input_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_y (int): Number of features for each time step in the input data.
        n_steps_out (int): Number of output steps.
        seed (int): Seed for random initialization of weights.
        kernel_size (int): Size of kernel in CNN.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    kernel_size = kernel_size

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=16,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
            name="input_layer",
            padding="SAME",
        )
    )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_1"))
    # model.add(
    #    layers.Conv1D(
    #        filters=16, kernel_size=kernel_size, activation="relu",
    #        name="conv1d_2", padding="SAME"
    #    )
    # )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_2"))
    # model.add(layers.Conv1D(filters=32, kernel_size=kernel_size,
    # activation="relu", name="conv1d_3"))
    model.add(layers.Flatten(name="flatten"))
    # model.add(layers.Dense(64, activation="relu", name="dense_2"))
    model.add(layers.Dense(32, activation="relu", name="dense_3"))
    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def cnn2(
    input_x,
    input_y,
    output_length=1,
    kernel_size=2,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a CNN model architecture using Keras.

    Args:
        input_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_y (int): Number of features for each time step in the input data.
        n_steps_out (int): Number of output steps.
        seed (int): Seed for random initialization of weights.
        kernel_size (int): Size of kernel in CNN.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    kernel_size = kernel_size

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=256,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
            name="input_layer",
            padding="SAME",
        )
    )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_1"))
    model.add(
        layers.Conv1D(
            filters=128, kernel_size=kernel_size, activation="relu", name="conv1d_1"
        )
    )
    model.add(
        layers.Conv1D(
            filters=64, kernel_size=kernel_size, activation="relu", name="conv1d_2"
        )
    )
    # model.add(layers.MaxPooling1D(pool_size=2, name="pool_1"))
    model.add(
        layers.Conv1D(
            filters=32, kernel_size=kernel_size, activation="relu", name="conv1d_3"
        )
    )
    # model.add(layers.Conv1D(filters=32, kernel_size=kernel_size,
    # activation="relu", name="conv1d_4"))
    # model.add(layers.Dropout(rate=0.1))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(128, activation="relu", name="dense_1"))
    model.add(layers.Dense(64, activation="relu", name="dense_2"))
    model.add(layers.Dense(32, activation="relu", name="dense_3"))
    # model.add(layers.Dropout(rate=0.1))
    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    # model.compile(optimizer=optimizers.Adam(lr=1e-8, beta_1=0.9, beta_2=0.999,
    #     epsilon=1e-8, decay=0.0001), loss=loss, metrics=metrics)

    return model


def dnn(
    input_x,
    output_length=1,
    seed=2020,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a DNN model architecture using Keras.

    Args:
        input_x (int): Number of features.
        output_length (int): Number of output steps.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    tf.random.set_seed(seed)

    model = models.Sequential()

    # One option after hp tuning
    # model.add(layers.Dense(160, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(96, activation="relu"))
    # model.add(layers.Dense(160, activation="relu"))

    # Another option after hp tuning
    # model.add(layers.Dense(400, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(336, activation="relu"))
    # model.add(layers.Dense(464, activation="relu"))
    # model.add(layers.Dense(272, activation="relu"))
    # model.add(layers.Dense(241, activation="relu"))
    # model.add(layers.Dense(64, activation="relu"))

    # Another option after hp tuning with window size 4
    # model.add(layers.Dense(88, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(880, activation="relu"))
    # model.add(layers.Dense(976, activation="relu"))
    # model.add(layers.Dense(848, activation="relu"))
    # model.add(layers.Dense(432, activation="relu"))
    # model.add(layers.Dense(752, activation="relu"))
    # model.add(layers.Dense(816, activation="relu"))
    # model.add(layers.Dense(944, activation="relu"))

    # Another option after hp tuning with window size 4
    model.add(layers.Dense(184, activation="relu", input_dim=input_x))
    model.add(layers.Dense(144, activation="relu"))
    model.add(layers.Dense(848, activation="relu"))
    model.add(layers.Dense(464, activation="relu"))
    model.add(layers.Dense(688, activation="relu"))
    model.add(layers.Dense(272, activation="relu"))

    # Leaky ReLU version
    # model.add(layers.Dense(400, input_dim=input_x))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dense(336))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dense(464))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dense(272))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dense(241))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dense(64))

    # Another option after hp tuning
    # model.add(layers.Dense(88, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(224, activation="relu"))
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(96, activation="relu"))
    # model.add(layers.Dense(896, activation="relu"))
    # model.add(layers.Dense(608, activation="relu"))
    # model.add(layers.Dense(656, activation="relu"))
    # model.add(layers.Dense(832, activation="relu"))
    # model.add(layers.Dense(784, activation="relu"))

    # Another option after hp tuning
    # model.add(layers.Dense(8, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(32, activation="relu"))
    # model.add(layers.Dense(208, activation="relu"))
    # model.add(layers.Dense(832, activation="relu"))
    # model.add(layers.Dense(992, activation="relu"))
    # model.add(layers.Dense(896, activation="relu"))

    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(output_length, activation=output_activation))

    # opt = tfk.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer=opt, loss=loss, metrics=metrics)

    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    # model.compile(optimizer="adam", loss="mean_squared_logarithmic_error", metrics=metrics)
    # model.compile(optimizer="adam", loss="cosine_similarity", metrics=metrics)

    return model


def dnn_simple(
    input_x,
    output_length=1,
    seed=2020,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a DNN model architecture using Keras.

    Args:
        input_x (int): Number of features.
        output_length (int): Number of output steps.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    tf.random.set_seed(seed)

    model = models.Sequential()

    # model.add(layers.Dense(2, activation="relu", input_dim=input_x))
    model.add(layers.Dense(8, activation="relu", input_dim=input_x))
    model.add(layers.Dense(32, activation="relu"))

    # model.add(layers.Dense(16, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(16, activation="relu"))
    # model.add(layers.Dense(8, activation="relu"))

    model.add(layers.Dense(output_length, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def lstm(
    hist_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        hist_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()

    """
    model.add(
        layers.LSTM(100, input_shape=(hist_size, n_features))
    )  # , return_sequences=True))
    model.add(layers.Dropout(0.5))
    # model.add(layers.LSTM(32, activation='relu'))
    # model.add(layers.LSTM(16, activation='relu'))
    model.add(layers.Dense(100, activation="relu"))
    """

    model.add(
        layers.LSTM(8, input_shape=(hist_size, n_features), recurrent_activation="relu")
    )  # , return_sequences=True))
    model.add(layers.Dropout(0.2))
    # model.add(layers.LSTM(32, activation='relu'))
    # model.add(layers.LSTM(16, activation='relu'))
    # model.add(layers.Dense(8, activation="relu"))
    # model.add(layers.Dense(8, activation="relu"))
    # model.add(layers.Dense(480, activation="relu"))

    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def lstm2(
    hist_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        hist_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()
    model.add(
        layers.LSTM(50, input_shape=(hist_size, n_features))
    )  # , return_sequences=True))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.LSTM(32, activation='relu'))
    # model.add(layers.LSTM(16, activation='relu'))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model

def rnn(
    hist_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        hist_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()

    model.add(layers.SimpleRNN(8, input_shape=(hist_size, n_features)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model

def gru(
    hist_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        hist_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()

    model.add(layers.GRU(8, input_shape=(hist_size, n_features)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model

class SequentialHyperModel(HyperModel):
    def __init__(self, input_x, input_y=0, n_steps_out=1):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out

    def build(self, hp, seed=2020):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.Dense(
                units=hp.Int(
                    name="units", min_value=2, max_value=16, step=2, default=8
                ),
                input_dim=self.input_x,
                activation="relu",
                name="input_layer",
            )
        )

        for i in range(hp.Int("num_dense_layers", min_value=0, max_value=4, default=1)):
            model.add(
                layers.Dense(
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=2,
                        max_value=16,
                        step=2,
                        default=8,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae", "mape"])

        return model

class LSTMHyperModel(HyperModel):
    def __init__(self, input_x, input_y=0, n_steps_out=1):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out

    def build(self, hp, seed=2020):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.LSTM(
                hp.Int(
                    name="lstm_units", min_value=4, max_value=256, step=8, default=128
                ),
                input_shape=(self.input_x, self.input_y),
            )
        )  # , return_sequences=True))

        add_dropout = hp.Boolean(name="dropout", default=False)

        if add_dropout:
            model.add(
                layers.Dropout(
                    hp.Float("dropout_rate", min_value=0.1, max_value=0.9, step=0.3)
                )
            )

        for i in range(hp.Int("num_dense_layers", min_value=1, max_value=4, default=2)):
            model.add(
                layers.Dense(
                    # units=64,
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=16,
                        max_value=512,
                        step=16,
                        default=64,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae", "mape"])

        return model

class CNNHyperModel(HyperModel):
    def __init__(self, input_x, input_y, n_steps_out=1):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out

    def build(self, hp, seed=2020):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.Conv1D(
                input_shape=(self.input_x, self.input_y),
                # filters=64,
                filters=hp.Int(
                    "filters", min_value=8, max_value=256, step=32, default=64
                ),
                # kernel_size=hp.Int(
                #     "kernel_size",
                #     min_value=2,
                #     max_value=6,
                #     step=2,
                #     default=4),
                kernel_size=2,
                activation="relu",
                name="input_layer",
                padding="same",
            )
        )

        for i in range(hp.Int("num_conv1d_layers", 1, 3, default=1)):
            model.add(
                layers.Conv1D(
                    # filters=64,
                    filters=hp.Int(
                        "filters_" + str(i),
                        min_value=8,
                        max_value=256,
                        step=32,
                        default=64,
                    ),
                    # kernel_size=hp.Int(
                    #     "kernel_size_" + str(i),
                    #     min_value=2,
                    #     max_value=6,
                    #     step=2,
                    #     default=4),
                    kernel_size=2,
                    activation="relu",
                    name=f"conv1d_{i}",
                )
            )

        # model.add(layers.MaxPooling1D(pool_size=2, name="pool_1"))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Flatten(name="flatten"))

        for i in range(hp.Int("num_dense_layers", min_value=1, max_value=8, default=2)):
            model.add(
                layers.Dense(
                    # units=64,
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=16,
                        max_value=1024,
                        step=16,
                        default=64,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae", "mape"])

        return model
