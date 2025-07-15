# Forest Cover Type Classification

This project aims to predict forest cover types solely from cartographic variables, without using remotely sensed data. The classification task identifies one of seven forest cover types for a given 30x30 meter cell using various geographical and environmental attributes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
  - [1.0 Exploratory Data Analysis (EDA)](#10-exploratory-data-analysis-eda)
  - [2.0 Data Preprocessing](#20-data-preprocessing)
  - [3.0 Model Training & Evaluation](#30-model-training--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)

## Project Overview

Understanding forest cover types is crucial for ecological studies, resource management, and conservation efforts. This project leverages a dataset of cartographic features derived from USGS and USFS data to classify forest cover types using machine learning algorithms. The study area focuses on four wilderness areas in the Roosevelt National Forest, northern Colorado, where forest types are primarily a result of ecological processes.

---

## Dataset

The dataset used is `covertype.csv`, consisting of cartographic variables and the forest `Cover_Type` (target). The data is in its raw form (not scaled) and includes both quantitative and binary qualitative features.

**Key Features:**

* **Quantitative**:
    * `Elevation` (meters)
    * `Aspect` (degrees azimuth)
    * `Slope` (degrees)
    * `Horizontal_Distance_To_Hydrology` (meters)
    * `Vertical_Distance_To_Hydrology` (meters)
    * `Horizontal_Distance_To_Roadways` (meters)
    * `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm` (0 to 255 index)
    * `Horizontal_Distance_To_Fire_Points` (meters)
* **Qualitative (Binary)**:
    * `Wilderness_Area` (4 binary columns: 0 or 1 for absence/presence)
    * `Soil_Type` (40 binary columns: 0 or 1 for absence/presence)
* **Target Variable**:
    * `Cover_Type` (integer 1 to 7, representing different forest types)

The dataset descriptions provide context for the wilderness areas (Neota, Rawah, Comanche Peak, Cache la Poudre) and their characteristic primary tree species (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Douglas-fir, Cottonwood/Willow, Aspen).

---

## Problem Statement

The core problem is a **multi-class classification task**: to accurately predict the `Cover_Type` (one of 7 classes) of a 30x30 meter forest cell based solely on the provided cartographic and environmental attributes.

---

## Methodology

### 1.0 Exploratory Data Analysis (EDA)

Initial steps involved:
* Loading the dataset and examining its basic structure.
* Generating descriptive statistics (`df.describe()`) to understand the distribution and range of each feature.
* Checking for missing values and data types.

### 2.0 Data Preprocessing

The raw data underwent several preprocessing steps to prepare it for machine learning models:
* **Feature Scaling**: Quantitative features were scaled using `StandardScaler` to normalize their ranges, preventing features with larger values from dominating the learning process.
* **Dimensionality Reduction (PCA)**: Principal Component Analysis (PCA) was applied to reduce the dimensionality of the feature space, especially beneficial given the large number of `Soil_Type` binary columns and potential correlations among features. This step transforms the original features into a smaller set of uncorrelated components while retaining most of the variance.

### 3.0 Model Training & Evaluation

The preprocessed data was split into training and testing sets. Two popular ensemble machine learning models were trained and evaluated:

* **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
* **XGBoost Classifier**: An optimized distributed gradient boosting library designed for speed and performance.

Models were evaluated primarily based on **accuracy** to assess their ability to correctly classify forest cover types.

---

## Results

After training and evaluating both models, **Random Forest Classifier achieved higher accuracy** than XGBoost Classifier in this specific context.

### Why Random Forest Outperformed XGBoost:

1.  **Data Size and Noise**: The dataset is large, clean, and tabular, which is highly conducive to Random Forests. XGBoost often excels in scenarios with noisy or highly imbalanced data, which was not the case here.
2.  **Overfitting Control**: Random Forest's **bagging** approach (training many trees on random subsets) intrinsically helps reduce variance and improves generalization. XGBoost, using **boosting**, can be more prone to overfitting if not carefully tuned, especially with many similar or redundant features.
3.  **Hyperparameter Sensitivity**: Random Forest is relatively robust to hyperparameter choices and frequently performs well with default or minimal tuning. XGBoost, conversely, is highly sensitive to tuning (e.g., learning rate, tree depth, subsample ratio), and sub-optimal tuning can lead to underperformance.

---

## Conclusion

This project successfully demonstrated the prediction of forest cover types using only cartographic variables. The **Random Forest Classifier proved to be the more effective model** for this dataset, outperforming XGBoost. This outcome highlights the importance of **experimenting with multiple models**, as sometimes simpler or less complex methods can yield superior results depending on the specific characteristics of the data and the level of hyperparameter tuning applied. The insights gained from this classification can support forest management and ecological studies.

---

## Technologies Used

* Python
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (for preprocessing, model selection, and evaluation)
* XGBoost
