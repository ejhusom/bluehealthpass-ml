#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze FitBit data.

Created:
    2021-10-26 Tuesday 11:09:52 

"""
import json
import os
from datetime import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES_LEVELS = ["deep", "wake", "light", "rem"]
CLASSIC_LEVELS = ["restless", "awake", "asleep"]


class FitBit:
    """Create a DataFrame containing processed FitBit data."""

    def __init__(
        self,
        filepath,
        directory_name_sleep="Sleep",
        directory_name_physical_activity="Physical Activity",
        directory_name_fas="FAS",
        fixed_fas_score=None,
    ):

        self.filepath = filepath
        self.filepath_sleep = self.filepath + directory_name_sleep
        self.filepath_sleep_scores = self.filepath_sleep + "/sleep_score.csv"
        self.filepath_physical_activity = (
            self.filepath + directory_name_physical_activity
        )
        self.filepath_fas = self.filepath + directory_name_fas
        self.fixed_fas_score = fixed_fas_score

        self.dfs = []

    def read_data(self):
        """Main function for reading all relevant data types."""

        self.read_sleep_score()
        self.read_sleep()
        self.read_hrv()
        self.read_timeseries("calories", self.filepath_physical_activity)
        self.read_timeseries("calories", self.filepath_physical_activity, sum_values=True)
        self.read_timeseries("distance", self.filepath_physical_activity)
        self.read_timeseries("distance", self.filepath_physical_activity, sum_values=True)
        self.read_timeseries("heart_rate", self.filepath_physical_activity)
        self.read_timeseries("steps", self.filepath_physical_activity, sum_values=True)
        self.read_timeseries(
            "lightly_active_minutes", self.filepath_physical_activity, sum_values=True
        )
        self.read_timeseries(
            "moderately_active_minutes",
            self.filepath_physical_activity,
            sum_values=True,
        )
        self.read_timeseries(
            "very_active_minutes", self.filepath_physical_activity, sum_values=True
        )
        self.read_timeseries(
            "sedentary_minutes", self.filepath_physical_activity, sum_values=True
        )
        self.read_stress_scores()
        self.read_profile()

        if self.fixed_fas_score == None:
            self.read_fas()
            self.combine()
        else:
            self.combine()
            self.df["fas"] = self.fixed_fas_score
            self.df["fas_category"] = self.fas2category(self.fixed_fas_score)

        self.df["age"] = self.age
        self.df["gender"] = self.gender
        self.df["weight"] = self.weight
        self.df["height"] = self.height
        self.df["bmi"] = self.bmi

        if self.gender == 0:
            s = -161
        else:
            s = 5

        self.df["bmr"] = 10 * self.weight + 6.25 * self.height - 5 * self.age + s

        self.df = self.df.fillna(0)

    def read_profile(self):
        """Read profile info."""

        profile = pd.read_csv(self.filepath + "Personal & Account/Profile.csv")
        self.age = datetime.now().year - int(profile["date_of_birth"][0][:4])
        self.gender = 0 if profile["gender"][0] == "FEMALE" else 1
        self.weight = profile["weight"][0]
        self.height = profile["height"][0]
        self.bmi = self.weight / ((self.height / 100) ** 2)

        # Uncomment to print info
        # print(self.age)
        # print(self.gender)
        # print(self.weight)
        # print(self.height)
        # print(self.bmi)
        # print("=========")

    def read_stress_scores(self):

        stress_scores = pd.read_csv(self.filepath + "Stress/Stress Score.csv")
        stress_scores["DATE"] = pd.to_datetime(stress_scores["DATE"])
        stress_scores.set_index("DATE", inplace=True)
        del stress_scores["UPDATED_AT"]
        del stress_scores["STATUS"]
        del stress_scores["CALCULATION_FAILED"]

        self.dfs.append(stress_scores)

    def read_sleep_score(self):

        sleep_scores = pd.read_csv(self.filepath_sleep_scores)
        sleep_scores["timestamp"] = pd.to_datetime(
            sleep_scores["timestamp"]
        ).dt.normalize()
        sleep_scores.set_index("timestamp", inplace=True)
        sleep_scores.index = sleep_scores.index.tz_convert(None)
        sleep_scores.sort_index(inplace=True)
        del sleep_scores["sleep_log_entry_id"]

        self.dfs.append(sleep_scores)

    def read_sleep(self):

        filepaths = find_files(self.filepath_sleep, prefix="sleep-")

        if not filepaths:
            return None

        sleep_dfs = []

        for filepath in filepaths:

            with open(filepath, "r") as infile:
                sleep_entry_json = json.load(infile)

            sleep_entry_df = pd.json_normalize(sleep_entry_json)
            sleep_dfs.append(sleep_entry_df)

        sleep_df = pd.concat(sleep_dfs)

        sleep_df["dateOfSleep"] = pd.to_datetime(sleep_df["dateOfSleep"])
        sleep_df.set_index("dateOfSleep", inplace=True)

        # Delete rows which does not contain main sleep
        sleep_df = sleep_df[sleep_df.mainSleep == True]

        self.dfs.append(sleep_df)

    def read_timeseries(self, name, filepath, sum_values=False):

        filepaths = find_files(filepath, prefix=name)

        if not filepaths:
            return None

        dfs = []

        for filepath in filepaths:

            with open(filepath, "r") as infile:
                json_data = json.load(infile)

            df = pd.json_normalize(json_data)
            dfs.append(df)

        df = pd.concat(dfs)

        # Change dateTime column to datetime type
        df["dateTime"] = pd.to_datetime(df["dateTime"])

        value_columns = [c for c in df.columns if c.startswith("value")]

        # Change value column to int type
        if len(value_columns) == 1:
            df[name] = df["value"].astype("float")
            del df["value"]
            new_value_columns = [name]
        else:
            new_value_columns = []
            for column in value_columns:
                new_column_name = name + "_" + column.split(".")[-1]
                df[new_column_name] = df[column].astype("float")
                del df[column]
                new_value_columns.append(new_column_name)

        df_resampled = df.resample("D", on="dateTime")
        df_new = pd.DataFrame()

        for column in new_value_columns:
            # Sum or mean values per day
            if sum_values:
                df_new[column] = df_resampled.sum()
            else:
                df_new[column + "_max"] = df_resampled[column].max()
                df_new[column + "_min"] = df_resampled[column].min()
                df_new[column + "_mean"] = df_resampled[column].mean()
                df_new[column + "_range"] = df_new[column + "_max"] - df_new[column + "_min"]
                df_new[column + "_std"] = df_resampled[column].std()

        self.dfs.append(df_new)

    def read_hrv(self):

        filepaths = find_files(self.filepath_sleep, prefix="Heart", extension="csv")

        if not filepaths:
            return None

        hrv_dfs = []

        for filepath in filepaths:
            if "Histogram" in filepath:
                continue

            hrv_df = pd.read_csv(filepath)
            hrv_dfs.append(hrv_df)

        hrv_df = pd.concat(hrv_dfs)

        # Change dateTime column to datetime type
        hrv_df["timestamp"] = pd.to_datetime(hrv_df["timestamp"])
        # Take mean value for each day
        hrv_df_resampled = pd.DataFrame()
        hrv_df_resampled["rmssd"] = hrv_df.resample("D", on="timestamp")["rmssd"].mean()
        hrv_df_resampled["coverage"] = hrv_df.resample("D", on="timestamp")[
            "coverage"
        ].mean()
        hrv_df_resampled["high_frequency"] = hrv_df.resample("D", on="timestamp")[
            "low_frequency"
        ].mean()
        hrv_df_resampled["low_frequency"] = hrv_df.resample("D", on="timestamp")[
            "high_frequency"
        ].mean()

        self.dfs.append(hrv_df_resampled)

    def read_estimated_oxygen_variation(self):
        pass

    def read_fas(self):

        filepaths = find_files(self.filepath_fas, prefix="fas")

        if not filepaths:
            return None

        fas_dfs = []

        for filepath in filepaths:
            fas_df = pd.read_csv(filepath)
            fas_dfs.append(fas_df)

        fas_scores = pd.concat(fas_dfs)
        fas_scores["date"] = pd.to_datetime(fas_scores["date"])
        fas_scores.set_index("date", inplace=True)

        self.dfs.append(fas_scores)

    def combine(self):

        self.df = self.dfs[0]

        for df in self.dfs[1:]:
            self.df = self.df.join(df)

        return self.df

    def plot_data(self):

        df = self.df.fillna(0)

        del df["levels.shortData"]
        del df["levels.data"]
        del df["startTime"]
        del df["endTime"]
        del df["type"]
        del df["mainSleep"]
        del df["fas_category"]

        print(df.info())
        pd.options.plotting.backend = "plotly"

        plt.figure()
        fig = df.plot()
        fig.show()
        # fig.write("test.html")

    def fas2category(self, fas):

        if fas < 22:
            return "no_fatigue"
        elif fas < 35:
            return "fatigue"
        else:
            return "extreme_fatigue"


def find_files(dir_path, prefix="", extension=""):
    """Find files in directory.

    Args:
        dir_path (str): Path to directory containing files.
        prefix (str): Only find files with a certain prefix. Default
            is an empty string, which means it will find all files.

    Returns:
        filepaths (list): All files found.

    """

    filepaths = []

    for f in sorted(os.listdir(dir_path)):
        if f.startswith(prefix) and f.endswith(extension):
            filepaths.append(dir_path + "/" + f)

    return filepaths

def plot_distributions(filepath):

    fas_scores = pd.read_csv(
        filepath + "/fatigue_score.csv",
        index_col=0,
    )

    ages = []
    fas = []
    male = 0
    female = 0

    for directory in os.listdir(filepath):
        if directory.startswith("patient"):
            print(f"Processing data in {directory}...")

            subject_number = directory.split("#")[-1]

            fb = FitBit(
                filepath=filepath + f"/{directory}/bhpbhp/",
                directory_name_sleep="Sleep",
                directory_name_physical_activity="Physical Activity",
                directory_name_fas="FAS",
                fixed_fas_score=fas_scores.loc[int(subject_number.lstrip("0"))][0],
            )

            fb.read_profile()
            ages.append(fb.age)
            fas.append(fas_scores.loc[int(subject_number.lstrip("0"))][0])

            if fb.gender == 0:
                female += 1
            else:
                male += 1

    ages = np.array(ages)

    print("Done!")
    print(f"Male: {male}")
    print(f"Female: {female}")

    print(f"Avg age: {np.mean(ages)} +- {np.std(ages)}")

    width = 4.5
    height = 2.7

    ax = plt.figure(figsize=(width,height)).gca()
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.hist(fas, bins=8)
    ax.set_xlabel("FAS-score")
    ax.set_ylabel("Frequency")
    ax.set_xlim([10, 50])
    plt.tight_layout()
    plt.savefig("fas_histogram.pdf")
    plt.show()

    ax = plt.figure(figsize=(width,height)).gca()
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.hist(ages)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("age_histogram.pdf")
    plt.show()


def restructure_and_save_data(filepath):

    fas_scores = pd.read_csv(
        filepath + "/fatigue_score.csv",
        index_col=0,
    )

    fas_scores_sorted = fas_scores.sort_values(by=" fatigue", inplace=False)

    n_subjects = len(fas_scores)

    data_set_indeces = {}

    first_idx = 0
    last_idx = n_subjects
    data_set_index = None

    # Giving suggestion for how to split/order dataset
    for i, f in enumerate(fas_scores_sorted.index):
        fas_score = fas_scores_sorted[" fatigue"].iloc[i]
        subject_index = f

        if i % 2:
            data_set_index = first_idx
            first_idx += 1
        else:
            data_set_index = last_idx
            last_idx -= 1

        data_set_indeces[f] = data_set_index

    dfs = []
    ages = []
    fas = []

    for directory in os.listdir(filepath):
        if directory.startswith("patient"):
            print(f"Processing data in {directory}...")

            subject_number = directory.split("#")[-1]
            if subject_number == "25":
                continue

            fb = FitBit(
                filepath=filepath + f"/{directory}/bhpbhp/",
                directory_name_sleep="Sleep",
                directory_name_physical_activity="Physical Activity",
                directory_name_fas="FAS",
                fixed_fas_score=fas_scores.loc[int(subject_number.lstrip("0"))][0],
            )

            fb.read_data()
            fb.df.to_csv(f"{str(data_set_indeces[int(subject_number)]).zfill(2)}_bhp_fitbit_{subject_number}_fas{int(fb.df['fas'][0])}.csv")
            # fb.df.to_csv(f"{data_set_indeces[int(subject_number)]}_bhp_fitbit_{subject_number}.csv")
            dfs.append(fb.df)
            # print(f" Gender: {fb.df['gender'][0]}")
            # print(f" Age: {fb.df['age'][0]}")
            print(f" FAS: {fb.df['fas'][0]}")

    print("Done!")

if __name__ == "__main__":

    filepath = sys.argv[1]

    restructure_and_save_data(filepath)
    plot_distributions(filepath)
