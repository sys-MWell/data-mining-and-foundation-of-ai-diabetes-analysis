

# Data Mining & Foundations of AI - Diabetes Prediction (Assessment 1)

## Project Overview
This project focuses on the early identification of diabetes risk factors using machine learning models. By analysing the healthcare diabetes dataset, we aim to explore various health indicators and their correlations with diabetes outcomes. The study highlights the use of machine learning algorithms like decision trees and random forests to predict diabetes risk and determine the most effective predictive model.

## Dataset
The project utilises the **Healthcare Diabetes Dataset**. You can download the dataset from the following link:
- [Healthcare Diabetes Dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)

### Dataset Features:
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / height in m²).
- **DiabetesPedigreeFunction**: Genetic score indicating the likelihood of diabetes.
- **Age**: Age in years.
- **Outcome**: Binary classification (1 = Diabetes, 0 = No Diabetes).

## Objective
To develop and validate a machine learning-based analytical model that accurately identifies diabetes risk factors in patients by analysing the relationships between health indicators in the dataset.

## Approach

### Data Preprocessing
- Identified missing or invalid data (e.g., "0" values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI).
- Handled missing data using **K-Nearest Neighbors (KNN) imputation** for better accuracy and data integrity.
- Excluded the **Insulin** column due to the significant proportion of missing data and potential biases.

### Exploratory Data Analysis (EDA)
1. **Correlation Matrix**:
   - Visualised relationships between variables.
   - Highlighted positive, negative, and no correlations.
   - Identified strong predictors for diabetes outcomes.

2. **Graphical Representations**:
   - **Histograms**: Showed distributions of attributes like Glucose, BloodPressure, BMI, and their relationship with diabetes outcomes.
   - **Scatter Plots**: Examined pairwise relationships between attributes (e.g., Glucose vs BMI, Age vs BloodPressure).

### Key Observations
- **Glucose vs Blood Pressure**: A positive relationship with diabetes outcomes, with higher glucose and blood pressure levels increasing diabetes risk.
- **Age vs BMI**: Older individuals with higher BMI values (30-40 range) were more likely to have diabetes.
- **Glucose vs BMI**: Glucose levels above 100 were strongly associated with positive diabetes outcomes.

### Machine Learning Models
The following machine learning algorithms were implemented and evaluated:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Gradient Boosting Classifier**

### Key Python Libraries Used
- `pandas`, `numpy`: Data manipulation and analysis.
- `matplotlib`, `seaborn`: Data visualisation.
- `sklearn`: Machine learning model training and evaluation.

## Results and Analysis
- **Missing Data Handling**: KNN imputation effectively filled gaps in critical variables like SkinThickness.
- **Model Performance**: Random Forest Classifier achieved the highest accuracy in predicting diabetes outcomes, outperforming other models.
- **Feature Importance**: Attributes such as Glucose, BMI, and Diabetes Pedigree Function were identified as key predictors of diabetes risk.

## Conclusion
This project demonstrates the potential of machine learning models in identifying diabetes risk factors. By addressing challenges like missing data and exploring feature relationships, the study provides a foundation for more advanced predictive modeling in healthcare.
