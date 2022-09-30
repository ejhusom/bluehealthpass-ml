#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot data frame.

Author:
    Erik Johannes Husom

Created:
    2022-05-18 onsdag 09:48:58 

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.options.plotting.backend = "plotly"

df = pd.read_csv(sys.argv[1])
print(df.info())

accepted_types = [
    "int64",
    "float64",
]

# Remove columns that does not contain numbers
for column in df.columns:
    if df[column].dtype not in accepted_types:
        del df[column]


scaler = MinMaxScaler()
X = scaler.fit_transform(df)
df = pd.DataFrame(X, columns=df.columns)

# Plot dataframe
fig = df.plot()
fig.show()
