# Breast Cancer Classification - Machine Learning Project

## Background Problem

In this project, the goal is to classify breast cancer tumors as **benign** (non-cancerous) or **malignant** (cancerous) using machine learning models. Early detection of breast cancer is essential for improving treatment outcomes and survival rates. By building a predictive model, the task is to distinguish between these two types of tumors using a dataset with various tumor characteristics. This project leverages the **Breast Cancer Wisconsin (Diagnostic)** dataset, which is widely used for evaluating classification algorithms in medical diagnosis.

The dataset includes features such as **mean radius**, **mean texture**, and other characteristics of the tumors, which help in classification. The dataset can be accessed via **scikit-learn**, or it can be downloaded directly through this [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

## Libraries & Version

This project uses the following libraries:

- **pandas v1.3.0** – for data manipulation and analysis.
- **numpy v1.21.0** – for numerical operations.
- **seaborn v0.11.1** – for data visualization.
- **scikit-learn v0.24** – for machine learning models and tools.
- **matplotlib v3.4.2** – for visualizations.

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic)** dataset, which contains 569 samples of tumors with 30 features for each sample. Each sample is labeled as either **benign** (0) or **malignant** (1). It is a well-known dataset for testing classification models, especially in medical diagnostics.

## Project Steps

1. **Import Libraries and Read Dataset**: Import necessary libraries and load the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Perform data exploration, visualizing the distribution of features and the target variable.
3. **Data Splitting**: Split the data into training and testing sets to evaluate model performance.
4. **Modeling**: Train several machine learning models, including **Logistic Regression**, **Naive Bayes**, **Random Forest**, **SVM**, **KNN**, and **Decision Tree**.
5. **Evaluation**: Evaluate the models using performance metrics such as accuracy, precision, recall, and F1-score.
6. **Visualization**: Use confusion matrices and other plots to compare the models' performance.

## Insights

Key insights from this project include:

1. **High accuracy across multiple models**: The **Naive Bayes** model achieved the highest accuracy of **97.37%**, followed by **Random Forest** at **96.49%** and **Logistic Regression** at **95.61%**.
2. **Model performance comparison**: The comparison of models showed that **Naive Bayes** was the most effective, while models like **KNN** and **Decision Tree** performed reasonably well with accuracies above **94%**.
3. **Balanced performance**: Models performed consistently well in terms of precision, recall, and F1-score, demonstrating their ability to distinguish between benign and malignant tumors effectively.
4. **Confusion Matrix**: Most models had very few misclassifications, with high correct predictions for both **malignant** and **benign** classes.

## Advice

This project can be further developed in the following areas:

1. **Hyperparameter Tuning**: Models like **Random Forest** and **SVM** can be further optimized by tuning hyperparameters such as **n_estimators**, **max_depth**, or **C** for better accuracy.
2. **Feature Engineering**: Additional features or feature selection techniques might improve model performance. Investigating which features have the most significant impact on classification could enhance the model.
3. **Deep Learning**: Exploring advanced models such as **neural networks** or **deep learning** techniques might lead to even higher accuracy in detecting malignant tumors.

## Conclusion

In this project, we built and evaluated various machine learning models to classify breast cancer tumors. The results show that Naive Bayes and Random Forest perform well in terms of accuracy, precision, and recall. The project also demonstrates the importance of **Exploratory Data Analysis (EDA)** in understanding data and preparing it for model training.

#EDA #python #machinelearning #breastcancerclassification
