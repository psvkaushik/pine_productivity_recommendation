
# 🌲 Pine Productivity Recommendation System

> A reproducible pipeline for predicting pine tree productivity and species classification.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Directory Structure](#dataset--directory-structure)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Data Preparation](#data-preparation)
5. [Usage](#usage)
   - [Task 1: Productivity Regression](#task-1-productivity-regression)
   - [Task 2: Species & Productivity Multi-Task](#task-2-species--productivity-multi-task)
6. [Configuration & Parameters](#configuration--parameters) 
7. [License](#license)

---

## Project Overview
This repository provides a two-stage pipeline to:

- **Task 1**: Predict pine tree productivity (regression problem).
- **Task 2**: Jointly classify pine species and predict productivity (multi-task learning).

By default, features like `latitude`, `longitude`, `date ranges`, and `TestId` are excluded as they show minimal predictive power.

---

## Project Structure

```
├── data/
│   └── ...
├── src/
│   ├── Task1/            # Regression scripts
│   │   ├── clustering.ipynb
│   │   └── Without_clustering.ipynb
│   └── Task2/            # Multi-task scripts
│       ├── mmmv.py
│       └── config.yaml
|       └── Versions_MTL_with_without_clustering.ipynb
└── README.md
```


---

## Dataset & Directory Structure

Please add the data into `data/` directory before running the code. The directory should look something like this:

```
data/
├── train_data.csv            
├── test_data.csv                  
├── split.ipynb        # Notebook to generate splits 
└── vis.ipynb          # Notebook for visualization
```

> **Note:** Do not alter the random seed in `split.ipynb` to ensure identical splits across runs.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.4+



### Installation

```bash
# Clone the repo
git clone https://github.com/psvkaushik/pine_productivity_recommendation.git
cd pine_productivity_recommendation

# Create virtual environment and install
conda create -n pine python=3.10
conda activate pine
pip install scikit-learn pandas tabulate numpy
```


### Data Preparation

1. Place CSVs in `data/`.
2. Run the split notebook:
   This generates `data/processed/train.csv`, `valid.csv`, and `test.csv`.

---

## Usage

### Task 1: Productivity Regression
Run the notebooks present in `src/Task1`


### Task 2: Species & Productivity Multi-Task
For Vanilla MTL run the `Versions_MTL_with_without_clustering.ipynb` in `src/Task2`
For the MTL with Mean and Varaince head run the following command
```bash
python src/Task2/mmmv.py 
```
**Output:** 
```
| Metric            |    Value |
|:------------------|---------:|
| Species Acc       | 0.781448 |
| Species Top-3 Acc | 0.977341 |
| Prod. MSE         | 2.52418  |
| Prod. MAE         | 1.17493  |
| Prod. R²          | 0.788992 |
| Prod. CRPS        | 0.829358 |
| 95% PICP          | 0.955784 |
| 95% MPIW          | 5.70942  |
```

---

## Configuration & Parameters

- All hyperparameters are defined in `config.yaml` per task.
- Key options:
  - `use_classifier_inputs`: bool (use the output from classifier head as input)
  - `hidden_dim`: int
  - `alpha`: float (confidence interval for Task 2)

---



## License
This project is released under the Apache License. See [LICENSE](LICENSE) for details.


