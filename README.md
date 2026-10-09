# House Price Prediction

## Objective
Predict house values from numerical housing features using a Linear Regression model.

## Dataset
Uses the California Housing dataset provided by scikit-learn. The dataset is fetched when the script runs, so an internet connection may be needed the first time.

## Requirements
Install the libraries from the repository root:

```bash
pip install -r requirements.txt
```

## Run
From the repository root:

```bash
python src/house_price_prediction.py
```

## Workflow
1. Load and explore housing data.
2. Split data into training and testing sets.
3. Standardize numerical features.
4. Train a Linear Regression model.
5. Evaluate with Mean Squared Error (MSE) and R².
6. Save an actual-vs-predicted chart and model metrics in `outputs/`.

## Technologies
Python, Pandas, Matplotlib, Scikit-learn.

## Note
The script creates output files when run successfully. The `outputs/` folder includes a `.gitkeep` file so the empty folder is retained in Git.
