#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:
    Erik Johannes Husom

Created:
    2020-09-16  

"""
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from joblib import dump
from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
    ridge_regression,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

import neural_networks as nn
from config import (
    DATA_PATH,
    DL_METHODS,
    MODELS_FILE_PATH,
    MODELS_PATH,
    NON_DL_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    TRAININGLOSS_PLOT_PATH,
)


def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    learning_method = params["learning_method"]
    use_early_stopping = params["early_stopping"]
    patience = params["patience"]
    target_size = yaml.safe_load(open("params.yaml"))["sequentialize"]["target_size"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]

    output_columns = np.array(pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)

    # Load training set
    train_data = np.load(filepath)

    X_train = train_data["X"]
    y_train = train_data["y"]

    n_features = X_train.shape[-1]

    hist_size = X_train.shape[-2]
    target_size = y_train.shape[-1]

    # Create sample weights
    sample_weights = np.ones_like(y_train)

    if params["weigh_samples"]:
        mask = (y_train < params["weight_max_threshold"]) & (
            y_train > params["weight_min_threshold"]
        )
        sample_weights[mask] = params["weight"]
        # sample_weights[y_train < params["weight_max_threshold"]] = params["weight"]

    if learning_method in DL_METHODS and params["hyperparameter_tuning"]:

        # In order to perform model tuning, any old model_tuning results omust
        # be deleted.
        if os.path.exists("model_tuning"):
            shutil.rmtree("model_tuning")

        if learning_method == "lstm":
            hypermodel = nn.LSTMHyperModel(hist_size, n_features)
        elif learning_method == "cnn":
            hypermodel = nn.CNNHyperModel(hist_size, n_features)
        else:
            hypermodel = nn.SequentialHyperModel(n_features)

        # hypermodel = nn.SequentialHyperModel(hist_size, n_features)
        hypermodel.build(HyperParameters())
        # hp = HyperParameters()
        # hp.Choice("num_layers", values=[1, 2])
        # hp.Fixed("kernel_size", value=4)
        # hp.Fixed("kernel_size_0", value=4)
        # tuner = Hyperband(
        #         hypermodel,
        #         # hyperparameters=hp,
        #         # tune_new_entries=True,
        #         objective="val_loss",
        #         # max_trials=10,
        #         # min_epochs=100,
        #         max_epochs=500,
        #         executions_per_trial=2,
        #         directory="model_tuning",
        #         project_name="BHP"
        # )
        tuner = BayesianOptimization(
            hypermodel,
            objective="val_loss",
            directory="model_tuning",
            project_name="BHP",
        )
        tuner.search_space_summary()
        tuner.search(
            X_train,
            y_train,
            epochs=params["n_epochs"],
            batch_size=params["batch_size"],
            validation_split=0.2,
            sample_weight=sample_weights,
        )
        tuner.results_summary()
        # best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        # model = tuner.hypermodel.build(best_hyperparameters)
        # print(model.summary())
        # history = model.fit(
        #     X_train, y_train,
        #     epochs=params["n_epochs"],
        #     batch_size=params["batch_size"],
        #     validation_split=0.2,
        #     sample_weight=sample_weights
        # )
        model = tuner.get_best_models()[0]
        print(model.summary())
        model.save(MODELS_FILE_PATH)

        return 0

    if classification:
        # if len(np.unique(y_train, axis=-1)) > 2:
        if onehot_encode_target:
            output_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        output_length = n_output_cols
        metrics = "accuracy"
        monitor_metric = "accuracy"
    else:
        output_activation = "linear"
        output_length = target_size
        loss = "mse"
        metrics = "mse"
        monitor_metric = "loss"

    # Build model
    if learning_method == "cnn":
        hist_size = X_train.shape[-2]
        model = nn.cnn(
            hist_size,
            n_features,
            output_length=output_length,
            kernel_size=params["kernel_size"],
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method == "cnn2":
        hist_size = X_train.shape[-2]
        model = nn.cnn2(
            hist_size,
            n_features,
            output_length=output_length,
            kernel_size=params["kernel_size"],
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method.startswith("dnn"):
        build_model = getattr(nn, learning_method)
        model = build_model(
            n_features,
            output_length=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method.startswith("lstm"):
        hist_size = X_train.shape[-2]
        build_model = getattr(nn, learning_method)
        model = build_model(
            hist_size,
            n_features,
            n_steps_out=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method.startswith("rnn"):
        hist_size = X_train.shape[-2]
        build_model = getattr(nn, learning_method)
        model = build_model(
            hist_size,
            n_features,
            n_steps_out=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method.startswith("gru"):
        hist_size = X_train.shape[-2]
        build_model = getattr(nn, learning_method)
        model = build_model(
            hist_size,
            n_features,
            n_steps_out=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif learning_method == "dt":
        if classification:
            model = DecisionTreeClassifier()
        else:
            if params["hyperparameter_tuning"]:
                dt_model = DecisionTreeRegressor()
                # model = GridSearchCV(
                model = RandomizedSearchCV(
                    dt_model,
                    {
                        "max_depth": [2, 5, 10, 15, 20, 50, 100],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 3, 5],
                    },
                    verbose=2,
                )
            else:
                model = DecisionTreeRegressor()
    elif learning_method == "rf":
        if classification:
            model = RandomForestClassifier()
        else:
            if params["hyperparameter_tuning"]:
                rf_model = RandomForestRegressor()
                # model = GridSearchCV(
                model = RandomizedSearchCV(
                    rf_model,
                    {
                        "max_depth": [2, 5, 10, 15, 20, 50, 100],
                        "n_estimators": [50, 100, 200, 400, 600, 800, 1000, 1200],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 3, 5],
                    },
                    verbose=2,
                )
            else:
                model = RandomForestRegressor()
    elif learning_method == "kneighbors" or learning_method == "kn":
        if classification:
            model = KNeighborsClassifier()
        else:
            if params["hyperparameter_tuning"]:
                kneighbors_model = KNeighborsRegressor()
                # model = GridSearchCV(
                model = RandomizedSearchCV(
                    kneighbors_model,
                    {
                        "n_neighbors": [2, 4, 5, 6, 10, 15, 20, 30],
                        "weights": ["uniform", "distance"],
                        "leaf_size": [10, 30, 50, 80, 100],
                        "algorithm": ["ball_tree", "kd_tree", "brute"],
                    },
                    verbose=2,
                )
            else:
                model = KNeighborsRegressor()
    elif learning_method == "gradientboosting" or learning_method == "gb":
        if classification:
            model = GradientBoostingClassifier()
        else:
            model = GradientBoostingRegressor()
    elif learning_method == "xgboost":
        if classification:
            model = xgb.XGBClassifier()
        else:
            # model = xgb.XGBRegressor(
            #         n_estimators=100,
            #         learning_rate=0.1,
            #         max_depth=10,
            #         max_leaves=2,
            # )
            if params["hyperparameter_tuning"]:
                xgb_model = xgb.XGBRegressor()
                # model = GridSearchCV(
                model = RandomizedSearchCV(
                    xgb_model,
                    {
                        "max_depth": [2, 5, 10, 15, 20, 50, 100],
                        "n_estimators": [50, 100, 200, 400, 600, 800, 1000, 1200],
                        "learning_rate": [0.3, 0.1, 0.001, 0.0001],
                    },
                    verbose=2,
                )
            else:
                model = xgb.XGBRegressor()
    elif learning_method == "mlp":
        if classification:
            model = MLPClassifier()
        else:
            model = MLPRegressor()
    elif learning_method == "linearregression":
        if classification:
            raise ValueError(
                f"Learning method {learning_method} only works with regression."
            )
        else:
            model = LinearRegression()
    elif learning_method == "ridgeregression":
        if classification:
            raise ValueError(
                f"Learning method {learning_method} only works with regression."
            )
        else:
            model = ridge_regression()
    elif learning_method == "lda":
        if classification:
            model = LinearDiscriminantAnalysis()
        else:
            raise ValueError(
                f"Learning method {learning_method} only works with classification."
            )
    elif learning_method == "sgd":
        if classification:
            model = SGDClassifier()
        else:
            model = SGDRegressor()
    elif learning_method == "qda":
        if classification:
            model = QuadraticDiscriminantAnalysis()
        else:
            raise ValueError(
                f"Learning method {learning_method} only works with classification."
            )
    elif learning_method == "svm":
        if classification:
            model = SVC()
        else:
            if params["hyperparameter_tuning"]:
                svm_model = SVR()
                # model = GridSearchCV(
                model = RandomizedSearchCV(
                    svm_model,
                    {
                        "kernel": ["linear", "poly", "rbf"],
                        "degree": [1, 3, 5],
                        "max_iter": [1, 5, 10],
                    },
                    verbose=2,
                )
            else:
                model = SVR()
    elif learning_method == "brnn":
        model = nn.brnn(
            data_size=X_train.shape[0],
            window_size=X_train.shape[1],
            feature_size=X_train.shape[2],
            batch_size=params["batch_size"],
            hidden_size=10,
        )  # TODO: Make this into a parameter
    elif learning_method == "bcnn":
        model = nn.bcnn(
            data_size=X_train.shape[0],
            window_size=X_train.shape[1],
            feature_size=X_train.shape[2],
            batch_size=params["batch_size"],
            kernel_size=params["kernel_size"],
            n_steps_out=target_size,
            output_activation=output_activation,
            classification=classification,
        )
    else:
        raise NotImplementedError(f"Learning method {learning_method} not implemented.")

    if learning_method in NON_DL_METHODS:
        model.fit(X_train, y_train)
        dump(model, MODELS_FILE_PATH)

        if params["hyperparameter_tuning"]:
            try:
                print(f"Best score CV: {model.best_score_}")
                print(f"Best params CV: {model.best_params_}")
            except:
                pass
    else:
        try:
            print(model.summary())
        except:
            pass

        # Save a plot of the model. Will not work if Graphviz is not installed, and
        # is therefore skipped if an error is thrown.
        try:
            PLOTS_PATH.mkdir(parents=True, exist_ok=True)
            plot_model(
                model,
                to_file=PLOTS_PATH / "model.png",
                show_shapes=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
            )
        except:
            print(
                "Failed saving plot of the network architecture, Graphviz must be installed to do that."
            )

        early_stopping = EarlyStopping(
            monitor="val_" + monitor_metric,
            patience=patience,
            verbose=4,
            restore_best_weights=True,
        )

        model_checkpoint = ModelCheckpoint(
            MODELS_FILE_PATH, monitor="val_" + monitor_metric  # , save_best_only=True
        )

        if use_early_stopping:
            # Train model for 10 epochs before adding early stopping
            history = model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=params["batch_size"],
                validation_split=0.25,
                sample_weight=sample_weights,
            )

            loss = history.history[monitor_metric]
            val_loss = history.history["val_" + monitor_metric]

            history = model.fit(
                X_train,
                y_train,
                epochs=params["n_epochs"],
                batch_size=params["batch_size"],
                validation_split=0.25,
                callbacks=[early_stopping, model_checkpoint],
                sample_weight=sample_weights,
            )

            loss += history.history[monitor_metric]
            val_loss += history.history["val_" + monitor_metric]

        else:
            history = model.fit(
                X_train,
                y_train,
                epochs=params["n_epochs"],
                batch_size=params["batch_size"],
                validation_split=0.25,
                sample_weight=sample_weights,
            )

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]

            model.save(MODELS_FILE_PATH)

        TRAININGLOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if classification:
            best_epoch = np.argmax(np.array(val_loss))
        else:
            best_epoch = np.argmin(np.array(val_loss))

        print(f"Best model in epoch: {best_epoch}")

        n_epochs = range(len(loss))

        plt.figure()
        plt.plot(n_epochs, loss, label="Training loss")
        plt.plot(n_epochs, val_loss, label="Validation loss")
        plt.legend()
        plt.savefig(TRAININGLOSS_PLOT_PATH)


if __name__ == "__main__":

    np.random.seed(2021)

    train(sys.argv[1])
