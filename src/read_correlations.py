#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read fas correlations.

Author:
    Erik Johannes Husom

Created:
    2022-05-30 mandag 11:29:41

"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open(sys.argv[1], "r") as f:
    c = json.load(f)

# c = dict(c)
# print(c)

# df = pd.DataFrame.from_dict(c)
# print(df)

var = []
val = []

for key in c:
    var.append(key)
    val.append(c[key])

df = pd.DataFrame(list(zip(var, val)), columns=["var", "val"])

df = df.dropna()

df = df.sort_values(by=["val"])

print(df.to_string())
