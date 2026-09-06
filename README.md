# Income-Level-Classification-Socioeconomic-Segmentation

## 👥 Team Members

* **Moataz Khalid** - [GitHub] | [LinkedIn](https://www.linkedin.com/in/moataz-abdou-770125424)
* **Omar Ahmed** - [GitHub](https://github.com/Omarahmed123458) | [LinkedIn](https://www.linkedin.com/in/omar-ahmed-hafez/)
* **Beshoy Shohdy** - [GitHub](https://github.com/beshoy-shohdy/beshoy-shohdy) | [LinkedIn](https://www.linkedin.com/in/beshoy-shohdy-2497b9282/)
* **Rana Elmaghraby** - [GitHub](https://github.com/RanaElmaghraby) | [LinkedIn](https://www.linkedin.com/in/rana-elmaghraby-8108a5366/)
  
**IncomeScope** is an end-to-end machine learning project developed as part of the **Samsung Innovation Campus (SIC) AI Track**.

The project focuses on predicting whether an individual earns **more than $50K or $50K or less annually** based on demographic, educational, occupational, and financial characteristics.

The project follows a complete machine learning workflow, starting from data quality assessment and exploratory data analysis, through feature engineering and preprocessing, to model development, evaluation, optimization, and deployment.

---

## 🎯 Problem Statement

Income level is an important socioeconomic indicator that can provide valuable insights into individuals and population segments.

The objective of this project is to develop a **binary classification model** capable of predicting whether an individual's annual income is:

* `<=50K`
* `>50K`

The model uses demographic, educational, occupational, and financial attributes when direct income information is unavailable.

### Potential Business Applications

The prediction can support:

* Customer segmentation
* Socioeconomic analysis
* Marketing analysis
* Product segmentation
* Business planning
* Financial customer profiling

---

## 📊 Dataset

The project uses the **Adult Income dataset**, containing demographic, employment, educational, and financial attributes.

The target variable is income level, making this a **binary classification problem**.

---

## 🧹 Data Cleaning

A detailed data quality assessment was performed before modeling.

### Main preprocessing steps

* Inspected missing values
* Replaced `"?"` values with missing values
* Applied mode imputation for categorical missing values
* Removed duplicate records
* Standardized categorical string values
* Identified redundant features
* Removed `fnlwgt` and redundant `education` information where appropriate
* Analyzed `capital-gain` and `capital-loss`

---

## 🛠️ Feature Engineering

Several features were engineered to improve the representation of the original data.

### Categorical Grouping

High-cardinality categorical variables were grouped into meaningful categories:

* **Workclass:** Private, Government, Self-employed, Other
* **Marital Status:** Single, Married, Previously Married
* **Occupation:** Professional, Office-Service, Skilled-Manual, Specialized
* **Relationship:** Spouse, Child, Other
* **Native Country:** US vs. Non-US

### Capital Features

Two additional features were created:

**`capital_net`**

Represents the net financial impact of capital gains and losses:

`capital_net = capital_gain - capital_loss`

**`has_capital_activity`**

A binary indicator showing whether an individual has any capital gain/loss activity.

---

## 🔎 Exploratory Data Analysis

EDA was performed to understand the target distribution, feature relationships, outliers, and income patterns.

### Key Findings

* The target variable is significantly imbalanced, with the lower-income class representing the majority.
* Individuals with capital activity showed a substantially higher proportion of high-income observations.
* Education level showed a strong relationship with income level.
* Higher-income individuals were more concentrated in higher education levels.
* Working hours showed a relationship with income, but working more hours alone did not guarantee higher income.
* Age showed an important relationship with income level.
* No strong multicollinearity was detected among the numerical features.
* Race and gender showed relatively low contribution compared with major predictive variables such as age, education, and working hours.

---

## 📈 Distribution & Transformation

Different transformation strategies were evaluated based on feature distributions.

### Yeo-Johnson Transformation

`capital_net` was highly right-skewed.

Its skewness was reduced from approximately **11.07 to 0.81** using a Yeo-Johnson transformation.

Yeo-Johnson was selected because it can handle both positive and negative values.

### Age

Age was approximately normally distributed with a relatively low proportion of outliers, so it was standardized without applying an aggressive power transformation.

### Hours per Week

`hours-per-week` was kept without a power transformation because transformations distorted its meaningful concentration around the standard working week.

---

## ⚖️ Class Imbalance

The dataset contains a significant class imbalance.

Since accuracy alone can be misleading in this problem, multiple evaluation metrics were considered:

* Precision
* Recall
* F1-score
* ROC-AUC

### Undersampling

Random undersampling was applied **only to the training data** after splitting the dataset.

This was done to prevent information leakage from the test set while reducing the dominance of the majority class during training.

---

## 🔄 Machine Learning Pipeline

The preprocessing and modeling workflow was organized into a reproducible pipeline.

### Pipeline

`Train/Test Split`

↓

`Undersampling`

↓

`Feature Transformation`

↓

`Yeo-Johnson Transformation`

↓

`Scaling`

↓

`One-Hot Encoding`

↓

`Machine Learning Model`

↓

`Evaluation`

This approach keeps preprocessing consistent and helps prevent data leakage.

---

# 🤖 Models

Several machine learning approaches were implemented and compared.

### 1. Random Forest

Used as a baseline ensemble model capable of capturing nonlinear relationships and interactions between features.

### 2. PCA + Random Forest

Principal Component Analysis was evaluated to determine whether dimensionality reduction could improve the Random Forest model.

### 3. XGBoost

A gradient boosting model was introduced to capture complex nonlinear relationships and improve classification performance.

### 4. Tuned XGBoost

GridSearchCV was used to investigate whether hyperparameter optimization could further improve the baseline XGBoost model.

---

## 🧪 Model Evaluation

To obtain a reliable estimate of model performance, **5-Fold Stratified Cross-Validation** was used.

Stratification ensures that the class distribution is maintained across folds.

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Because the dataset is imbalanced, particular attention was given to **Recall, F1-score, and ROC-AUC** rather than relying on accuracy alone.

---

# 🏆 Results

### Best Performing Model: XGBoost

The XGBoost model achieved the strongest overall performance.

| Metric   |    XGBoost |
| -------- | ---------: |
| Recall   | **89.76%** |
| F1-Score | **86.04%** |
| ROC-AUC  | **92.32%** |

XGBoost outperformed the Random Forest models overall.

---

## 🧩 PCA Analysis

PCA did not improve Random Forest performance.

The baseline Random Forest achieved approximately **81% accuracy**, while the PCA + Random Forest approach dropped to approximately **80%**.

This suggests that PCA was not particularly beneficial for this tree-based model.

One reason is that PCA creates linear combinations of the original variables, which can reduce the direct interpretability of features such as `age` and `education-num` while not necessarily benefiting tree-based decision boundaries.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was applied to XGBoost.

However, tuning produced almost identical performance to the baseline configuration:

* Baseline XGBoost ROC-AUC: **92.32%**
* Tuned XGBoost ROC-AUC: **92.30%**

This suggests that the baseline configuration was already close to optimal for the selected search space, or that the search grid was not broad enough to produce meaningful improvement.

---

## 📊 Key Insights

The modeling and EDA stages highlighted several important patterns:

1. **Age, education, and working hours are major predictive factors.**
2. **Capital activity is strongly associated with higher income levels.**
3. **Education level shows a clear shift toward higher income categories.**
4. **Working hours alone are not sufficient to explain income level.**
5. **PCA did not improve the tree-based model.**
6. **XGBoost provided the strongest overall predictive performance.**
7. **Hyperparameter tuning provided negligible improvement over baseline XGBoost.**

---

## 📊 Dashboard
https://drive.google.com/file/d/1Yv_zlH7lZtfzIesttKAAanH6ebTGb1T7/view?usp=sharing

An interactive dashboard was developed to visualize:

* Income distribution
* Demographic patterns
* Education vs. income
* Capital activity
* Income-related insights
* Model prediction results

---

## 🚀 Deployment
https://annual-income-classifier.streamlit.app

The final machine learning solution was integrated into an application that allows users to provide individual characteristics and receive an income-level prediction.

**Prediction Output:**

* `<=50K`
* `>50K`

---

## ⚠️ Limitations

### Class Balancing

Random undersampling changes the natural class distribution and removes some majority-class observations, which may cause information loss.

### Generalizability

The dataset represents a specific population and historical context, so model performance may not directly generalize to other countries, populations, or time periods.

### Fairness Considerations

Income prediction is a socioeconomic task, and demographic variables may introduce fairness concerns. Model performance should therefore be evaluated across different population groups before real-world deployment.

---

## 🔮 Future Work

Potential improvements include:

* Comparing undersampling with SMOTE and class-weighted learning
* Training on more diverse and recent datasets
* Exploring ensemble methods and advanced boosting algorithms
* Performing more extensive hyperparameter optimization
* Evaluating calibration and decision thresholds
* Performing subgroup fairness analysis
* Testing the model on external datasets
* Improving the deployed application and monitoring model performance over time

---

## 👥 Team

**Samsung Innovation Campus — AI Track**

* Rana Elmaghraby
* Moataz Khalid
* Omar Ahmed
* Beshoy Shohdy

---

## 🧰 Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit` · `Jupyter Notebook`

---

## 📌 Project Type

**Machine Learning | Binary Classification | Supervised Learning | End-to-End ML Project**

Developed as part of the **Samsung Innovation Campus AI Track**.
