# A Comparative Analysis of Forecasting Models for Macroeconomic Time Series

This project develops an end-to-end forecasting system to predict U.S. Total Vehicle Sales. It rigorously benchmarks three distinct modeling paradigms: a traditional statistical model (SARIMAX), an industry-standard machine learning model (LightGBM), and a state-of-the-art deep learning foundational model (TimesFM).

## Contribution to TimesFM Implementation

A significant challenge in this project was that the publicly available source code for the TimesFM model was not directly runnable for this use case and contained a critical bug.

* **Problem:** The original `model.py` script failed during the forecasting step with an `AssertionError`, indicating a tensor with an incorrect 4D shape was being passed to the Transformer encoder, which expects a 3D tensor.
* **Analysis:** I traced the error to the `PatchEmbedding` class, where the tensor was correctly projected but not properly reshaped, leaving a redundant dimension.
* **Solution:** I corrected the bug by adding a `.squeeze(2)` operation within the `PatchEmbedding.forward` method. This fix correctly reshapes the tensor to the 3D format required by the model's encoder, resolving the error.

The corrected and runnable version of `model.py` is included in this repository, making this implementation of TimesFM accessible for similar forecasting tasks.

## 1. Business Problem
The goal of this project is to provide a data-driven forecast of vehicle sales to inform strategic planning (e.g., production schedules, marketing budgets) for a major U.S. automotive company. The primary question is: "Which forecasting methodology provides the most accurate prediction for future sales given the current economic climate?"

## 2. Data
The dataset consists of 12 monthly macroeconomic indicators sourced from the Federal Reserve Economic Data (FRED) repository, spanning from 1993 to the present. The target variable for the forecast is `Total_Vehicle_Sales`.

## 3. Methodology
The project followed a three-step process:
1.  **Exploratory Data Analysis (EDA):** Analyzed historical trends, seasonality, and correlations. A key finding was the non-stationary nature of the data, which informed the modeling approach.
2.  **Comparative Modeling:** Developed and trained three multivariate models to forecast vehicle sales using other economic indicators as features:
    * **SARIMAX:** A traditional statistical model for time series.
    * **LightGBM:** A gradient-boosting machine learning model requiring extensive feature engineering (lags, date features).
    * **TimesFM:** A deep learning foundational model.
3.  **Evaluation:** Compared the models based on their Root Mean Squared Error (RMSE) on a hold-out test set.

## 4. Results
The models were benchmarked, and the final performance was as follows:

*(Insert the final comparison chart image here)*

| Model    | RMSE     |
|----------|----------|
| TimesFM  | 0.876361 |
| SARIMAX  | 0.901941 |
| LightGBM | 1.004777 |

**Conclusion:** The TimesFM model delivered the most accurate forecast, demonstrating the potential of foundational models to capture complex patterns in economic data more effectively than traditional or standard machine learning methods.

## 5. How to Run
1.  Clone this repository.
2.  Install the required libraries: `pip install -r requirements.txt`
3.  Run the Jupyter Notebooks `01_EDA.ipynb` and `02_comparative_modeling.ipynb`.

