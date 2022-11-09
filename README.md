# Machine learning pipeline for BlueHealthPass

Estimating Fatigue Assessment Score (FAS) using machine learning.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.


## Usage


Basic steps to produce a model:

1. Restructure raw data:

```
```

2. Place the data files (output of previous step) in `assets/data/raw`.
3. Configure parameters in `params.yaml`.
4. Run:

```
dvc repro
```
