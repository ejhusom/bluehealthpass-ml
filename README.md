# Machine learning pipeline for BlueHealthPass

Estimating Fatigue Assessment Score (FAS) using machine learning.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.

## Installation

1. Download the source code using `git`: [github.com/ejhusom/bluehealthpass-ml](https://github.com/ejhusom/bluehealthpass-ml)

```
git clone https://github.com/ejhusom/bluehealthpass-ml.git
```

2. Installing required Python-packages:

```
cd bluehealthpass-ml
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

3. Set up DVC:

```
dvc init --no-scm
```

## Usage


Basic steps to produce a model:

1. Restructure raw data:

```
python3 src/fitbit.py [path/to/fitbit/data]
```

2. Place the data files (output of previous step) in `assets/data/raw`.
3. Configure parameters in `params.yaml`.
4. Run:

```
dvc repro
```
