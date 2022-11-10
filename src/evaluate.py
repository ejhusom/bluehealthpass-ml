#!/usr/bin/env python3
"""Evaluate deep learning model.

Author:
    Erik Johannes Husom

Created:
    2020-09-17

"""
import json
import shutil
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sn
import tensorflow as tf
import yaml
from joblib import load
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import metrics, models

import neural_networks as nn
from config import (
    DATA_PATH,
    INPUT_FEATURES_PATH,
    INPUT_SCALER_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
)


def evaluate(model_filepath, train_filepath, test_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        train_filepath (str): Path to train set.
        test_filepath (str): Path to test set.

    """

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    params_split = yaml.safe_load(open("params.yaml"))["split"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]
    show_inputs = params["show_inputs"]
    learning_method = params_train["learning_method"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    if show_inputs:
        inputs = X_test
    else:
        inputs = None

    PREDICTIONS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_test).to_csv(PREDICTIONS_PATH / "true_values.csv")

    # pandas data frame to store predictions and ground truth.
    df_predictions = None

    y_pred = None

    if learning_method in NON_DL_METHODS:
        model = load(model_filepath)
        y_pred = model.predict(X_test)
    elif learning_method == "brnn":
        model = nn.brnn(
            data_size=X_test.shape[0],
            window_size=X_test.shape[1],
            feature_size=X_test.shape[2],
            # hidden_size=params_train["hidden_size"],
            batch_size=params_train["batch_size"],
        )
        model.load_weights(model_filepath)
        signal_start = 1000
        signal_end = 500

        y_pred = model(X_test)
        assert isinstance(y_pred, tfd.Distribution)

        mean = y_pred.mean().numpy()

        aleatoric, epistemic = compute_uncertainty(
            model=model, test_data=X_test, iterations=200
        )

        total_unc = np.sqrt(aleatoric**2 + epistemic**2)

        prediction_interval_plot(
            true_data=y_test[:, -1],
            predicted_mean=mean,
            predicted_std=total_unc,
            plot_path=PLOTS_PATH,
            file_name="confidence_plot.html",
            experiment_length=len(X_test),
        )
    elif learning_method == "bcnn":

        model = nn.bcnn(
            data_size=X_test.shape[0],
            window_size=X_test.shape[1],
            feature_size=X_test.shape[2],
            batch_size=params_train["batch_size"],
            kernel_size=params_train["kernel_size"],
        )
        model.load_weights(model_filepath)

        input_columns = pd.read_csv(INPUT_FEATURES_PATH).values.tolist()[0][1:]
        X_test[300:305, :, :] = 3
        y_pred = model(X_test)

        assert isinstance(y_pred, tfd.Distribution)
        mean = y_pred.mean().numpy()

        aleatoric, epistemic = compute_uncertainty(
            model=model, test_data=X_test, iterations=100
        )

        # uncertainties can be accurately predicted by the superposition of these uncertainties
        total_unc = np.sqrt(aleatoric**2 + epistemic**2)
        prediction_interval_plot(
            true_data=y_test[:, 0],
            predicted_mean=mean,
            predicted_std=total_unc,
            plot_path=PLOTS_PATH,
            file_name="confidence_plot.html",
            experiment_length=len(X_test),
        )

        # Use the training data for deep explainer => can use fewer instances
        # Fit the normalizer.
    else:
        model = models.load_model(model_filepath)
        y_pred = model.predict(X_test)

    if onehot_encode_target:
        y_pred = np.argmax(y_pred, axis=-1)
    elif classification:
        y_pred = np.array((y_pred > 0.5), dtype=np.int)

    if classification:

        if onehot_encode_target:
            y_test = np.argmax(y_test, axis=-1)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        plot_prediction(y_test, y_pred, info="Accuracy: {})".format(accuracy))

        plot_confusion(y_test, y_pred)

        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(dict(accuracy=accuracy), f)

        # ==========================================
        # TODO: Fix SHAP code
        # explainer = shap.TreeExplainer(model, X_test[:10])
        # shap_values = explainer.shap_values(X_test[:10])
        # plt.figure()
        # shap.summary_plot(shap_values[0][:,0,:], X_test[:10][:,0,:])
        # shap.image_plot([shap_values[i][0] for i in range(len(shap_values))], X_test[:10])
        # input_columns = pd.read_csv(INPUT_FEATURES_PATH).iloc[:,-1]
        # print(input_columns)
        # shap.force_plot(explainer.expected_value[0], shap_values[0][0])

        # plt.savefig("test.png")

        # feature_importances = model.feature_importances_
        # imp = list()
        # for i, f in enumerate(feature_importances):
        #     imp.append((f,i))

        # sorted_feature_importances = sorted(imp)

        # print("Feature importances")
        # print(sorted_feature_importances)
        # ==========================================

    # Regression:
    else:

        if learning_method == "bcnn" or learning_method == "brnn":
            mse = mean_squared_error(y_test[:, -1], mean[:, -1])
            rmse = mean_squared_error(y_test[:, -1], mean[:, -1], squared=False)
            mape = mean_absolute_percentage_error(y_test[:, -1], mean[:, -1])
            r2 = r2_score(y_test[:, -1], mean[:, -1])

            plot_prediction(
                y_test[:, -1], mean[:, -1], inputs=inputs, info="(R2: {})".format(r2)
            )
            plot_true_vs_pred(y_test[:, -1], mean[:, -1])
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            plot_prediction(y_test, y_pred, inputs=inputs, info=f"(R2: {r2:.2f})")
            plot_true_vs_pred(y_test, y_pred, inputs=inputs)
            regression2classification(y_test, y_pred)

        print("MSE: {}".format(mse))
        print("RMSE: {}".format(rmse))
        print("MAPE: {}".format(mape))
        print("R2: {}".format(r2))

        # Only plot predicted sequences if the output samples are sequences.
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            plot_sequence_predictions(y_test, y_pred)

        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(dict(mse=mse, rmse=rmse, mape=mape, r2=r2), f)

    # Print feature importances of the ML algorithm supports it.
    try:
        feature_importances = model.feature_importances_
        imp = list()
        for i, f in enumerate(feature_importances):
            imp.append((f, i))

        sorted_feature_importances = sorted(imp)[::-1]
        input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

        print("-------------------------")
        print("Feature importances:")

        for i in range(len(sorted_feature_importances)):
            print(
                f"Feature: {input_columns.iloc[i,0]}. Importance: {feature_importances[i]:.2f}"
            )

        print("-------------------------")
    except:
        pass

    save_predictions(pd.DataFrame(y_pred))


def compute_uncertainty(model, test_data, iterations=100):
    """A function to compute aleatoric and epistemic uncertainty of a probabilistic Gaussain model
    based on Ensemble method

    Args:
        model: Probabilistic gaussian model
        test_data: test dataset
        iterations: number of iteration

    Returns: (tuple of of ndarray) representing aleatoric and epistemic uncertainty of the model

    """
    means = []
    stddevs = []
    for _ in range(iterations):
        means.append(model(test_data).mean().numpy())
        stddevs.append(model(test_data).stddev().numpy())

    means = np.concatenate(means, axis=1)
    stddevs = np.concatenate(stddevs, axis=1)
    overall_mean = np.mean(means, axis=1)

    aleatoric = np.mean(stddevs, axis=1)
    epistemic = np.sqrt(np.mean(means**2, axis=1) - overall_mean**2)

    return aleatoric, epistemic


def plot_confusion(y_test, y_pred):
    """Plotting confusion matrix of a classification model."""

    output_columns = np.array(pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)
    indeces = np.arange(0, n_output_cols, 1)

    confusion = confusion_matrix(y_test, y_pred, normalize="true")
    # labels=indeces)

    print(confusion)

    df_confusion = pd.DataFrame(confusion)

    df_confusion.index.name = "True"
    df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_confusion, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.savefig(PLOTS_PATH / "confusion_matrix.png")


def save_predictions(df_predictions):
    """Save the predictions along with the ground truth as a csv file.

    Args:
        df_predictions_true (pandas dataframe): pandas data frame with the predictions and ground truth values.

    """

    PREDICTIONS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_predictions.to_csv(PREDICTIONS_FILE_PATH, index=False)


def regression2classification(y_true, y_pred):

    n = len(y_true)

    misclassifications = 0

    for t, p in zip(y_true, y_pred):

        if fas2category(t) != fas2category(p):
            misclassifications += 1

    print(f"Number of misclassifications: {misclassifications}")
    print(f"Percentage of misclassifications: {misclassifications/n:.2f}")
    print(f"Accuracy: {1 - misclassifications/n:.2f}")


def fas2category(fas):

    if fas < 22:
        return 0
    elif fas < 35:
        return 1
    else:
        return 2


def plot_true_vs_pred(y_true, y_pred, inputs=None):

    if len(y_true.shape) > 1:
        y_true = y_true[:, -1].reshape(-1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred[:, -1].reshape(-1)

    plt.figure(figsize=(4.5, 4.5))
    plt.xlabel("Actual FAS-score")
    plt.ylabel("Predicted FAS-score")
    plt.xlim([min(10, np.min(y_true)), 50])
    plt.ylim([min(10, np.min(y_pred)), 50])
    plt.plot([10, 50], [10, 50], c="gray", alpha=0.5, label="perfect prediction")

    # Add shapes indicating categories
    # no_fatigue_shape = plt.Rectangle((10,10), 12, 12, fc="green", alpha=0.2)
    # fatigue_shape = plt.Rectangle((22,22), 13, 13, fc="orange", alpha=0.2)
    # extreme_fatigue_shape = plt.Rectangle((35,35), 15, 15, fc="red", alpha=0.2)
    # plt.gca().add_patch(no_fatigue_shape)
    # plt.gca().add_patch(fatigue_shape)
    # plt.gca().add_patch(extreme_fatigue_shape)

    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.legend()
    plt.savefig(PLOTS_PATH / "true_vs_pred.png")

    input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

    new_inputs = []

    if len(inputs.shape) == 3:
        n_features = inputs.shape[-1]
    elif len(inputs.shape) == 2:
        n_features = len(input_columns)

    for i in range(n_features):

        if len(inputs.shape) == 3:
            new_inputs.append(inputs[:, -1, i])
        elif len(inputs.shape) == 2:
            new_inputs.append(inputs[:, i - n_features])

    new_inputs = np.transpose(np.array(new_inputs))
    inputs = new_inputs

    input_scaler = joblib.load(INPUT_SCALER_PATH)
    inputs = input_scaler.inverse_transform(inputs)

    age_index = input_columns.index[input_columns[0] == "age"]
    gender_index = input_columns.index[input_columns[0] == "gender"]
    weight_index = input_columns.index[input_columns[0] == "weight"]

    age = inputs[:, age_index].flatten()
    gender = inputs[:, gender_index].flatten()
    weight = inputs[:, weight_index].flatten()

    zipped = list(zip(y_true, y_pred, age, gender, weight))
    df = pd.DataFrame(zipped, columns=["y_true", "y_pred", "age", "gender", "weight"])

    hovertext = []

    for i in range(len(df)):
        hovertext.append(
            f"Age: {df['age'].iloc[i]}, Gender: {df['gender'].iloc[i]}, Weight: {df['weight'].iloc[i]}"
        )

    df["hovertext"] = hovertext

    # fig = px.scatter(df, x="y_true", y="y_pred", hover_data=["age", "gender", "weight"])
    # fig.add_line(pd.DataFrame([[10,50],[10,50]], columns=["x","y"]), x="x", y="y")

    fig = make_subplots()

    fig.add_trace(go.Scatter(x=[10, 50], y=[10, 50], marker=dict(opacity=0.3)))

    fig.add_trace(
        go.Scatter(
            x=df["y_true"],
            y=df["y_pred"],
            mode="markers",
            marker=dict(
                color=df["age"], size=df["weight"] / 8, symbol=df["gender"], opacity=0.4
            ),
            hovertext=df["hovertext"],
        )
    )

    fig.write_html(str(PLOTS_PATH / "true_vs_pred.html"))


def plot_prediction(y_true, y_pred, inputs=None, info=""):
    """Plot the prediction compared to the true targets.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.
        inputs (array): Inputs corresponding to the targets passed. If
            provided, the inputs will be plotted together with the targets.
        info (str): Information to include in the title string.

    """

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(y_true.shape) > 1:
        y_true = y_true[:, -1].reshape(-1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred[:, -1].reshape(-1)

    fig.add_trace(
        go.Scatter(x=x, y=y_true, name="true"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x, y=y_pred, name="pred"),
        secondary_y=False,
    )

    if inputs is not None:
        input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

        if len(inputs.shape) == 3:
            n_features = inputs.shape[-1]
        elif len(inputs.shape) == 2:
            n_features = len(input_columns)

        for i in range(n_features):

            if len(inputs.shape) == 3:
                fig.add_trace(
                    go.Scatter(x=x, y=inputs[:, -1, i], name=input_columns.iloc[i, 0]),
                    secondary_y=True,
                )
            elif len(inputs.shape) == 2:
                fig.add_trace(
                    go.Scatter(
                        x=x, y=inputs[:, i - n_features], name=input_columns.iloc[i, 0]
                    ),
                    secondary_y=True,
                )

    fig.update_layout(title_text="True vs pred " + info)
    fig.update_xaxes(title_text="time step")
    fig.update_yaxes(title_text="target unit", secondary_y=False)
    fig.update_yaxes(title_text="scaled units", secondary_y=True)

    fig.write_html(str(PLOTS_PATH / "prediction.html"))

    # fig.update_traces(line=dict(width=0.8))
    # fig.write_image("plot.pdf", height=270, width=560)
    # fig.write_image("plot.png", height=270, width=560, scale=10)

    return fig


def plot_sequence_predictions(y_true, y_pred):
    """
    Plot the prediction compared to the true targets.

    """

    target_size = y_true.shape[-1]
    pred_curve_step = target_size

    pred_curve_idcs = np.arange(0, y_true.shape[0], pred_curve_step)
    # y_indeces = np.arange(0, y_true.shape[0]-1, 1)
    y_indeces = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])

    n_pred_curves = len(pred_curve_idcs)

    fig = go.Figure()

    y_true_df = pd.DataFrame(y_true[:, 0])

    fig.add_trace(go.Scatter(x=y_indeces, y=y_true[:, 0].reshape(-1), name="true"))

    predictions = []

    for i in pred_curve_idcs:
        indeces = y_indeces[i : i + target_size]

        if len(indeces) < target_size:
            break

        y_pred_df = pd.DataFrame(y_pred[i, :], index=indeces)

        predictions.append(y_pred_df)

        fig.add_trace(
            go.Scatter(
                x=indeces, y=y_pred[i, :].reshape(-1), showlegend=False, mode="lines"
            )
        )

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(PLOTS_PATH / "prediction_sequences.html"))


def prediction_interval_plot(
    true_data,
    predicted_mean,
    predicted_std,
    plot_path,
    file_name="Uncertainty_plot.html",
    experiment_length=5000,
):
    """This function plots 95% prediction interval for bayesian neural network with gaussain output

    Args:
        true_data:  (ndarray) label of test dataset
        predicted_mean:  (ndarray) mean of predicted test data
        predicted_std:  (ndarray) standerd deviation of predicted test data
        plot_path: (path object) path to where the plot to be saved
        file_name: (str) name of saved plot
        experiment_length: length of plot along x-axis

    Returns: None

    """
    x_idx = np.arange(0, experiment_length, 1)
    fig = go.Figure(
        [
            go.Scatter(
                name="Original measurement",
                x=x_idx,
                y=np.squeeze(true_data[0:experiment_length]),
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
            go.Scatter(
                name="predicted values ",
                x=x_idx,
                y=np.squeeze(predicted_mean[0:experiment_length]),
                mode="lines",
                # marker=dict(color="#444"),
                line=dict(color="rgb(255, 10, 10)"),
                showlegend=True,
            ),
            go.Scatter(
                name="95% prediction interval",
                x=x_idx,
                y=np.squeeze(predicted_mean[0:experiment_length])
                - 1.96 * predicted_std[0:experiment_length],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="95% prediction interval",
                x=x_idx,
                y=np.squeeze(predicted_mean[0:experiment_length])
                + 1.96 * predicted_std[0:experiment_length],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=True,
            ),
        ]
    )
    fig.update_layout(
        yaxis_title="target vs predicted", title="Uncertainty estimation", hovermode="x"
    )
    # fig.update_yaxes(range=[-10, 10])
    # fig.update(layout_yaxis_range=[-5, 5])
    # fig.show()

    if plot_path is not None:
        fig.write_html(str(plot_path / file_name))

if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            evaluate(
                "assets/models/model.h5",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz",
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        evaluate(sys.argv[1], sys.argv[2], sys.argv[3])


