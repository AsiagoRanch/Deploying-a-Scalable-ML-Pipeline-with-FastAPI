# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Developed by: Nathanael Vasquez
- Model date: September 20, 2025
- Model version: 1.0
- Model type: Logistic Regression
- Training Data: UCI Adult Census Income Dataset (1994)
- Information: This model is part of a project to demonstrate building and deploying a machine learning pipeline.

## Intended Use

**Primary intended use:** This model is intended for educational purposes to predict whether an individual's annual income exceeded $50,000 based on 1994 US census data. It serves as a practical example for MLOps concepts like automated testing, deployment, and performance monitoring on data slices.

**Out-of-scope uses:** This model is not suitable for any real-world application. It must not be used to make decisions about employment, credit, housing, or any other real-world outcome. The data is outdated and the model may contain significant historical and societal biases.

## Training Data

The model was trained on the UCI Census Income dataset, which contains 32,561 records from the 1994 US Census database. The dataset includes a mix of numerical and categorical features describing individual demographics, such as age, workclass, education, race, and sex. The target variable is salary, indicating whether income is <=50K or >50K.

## Evaluation Data

The model was evaluated on a 20% holdout test set that was separated from the original data before training. The split was stratified based on the salary column to ensure that both the training and testing sets have the same proportion of high and low-income individuals.

## Metrics

The model's performance was evaluated using Precision, Recall, and F1-score. These metrics were chosen over simple accuracy because the dataset is imbalanced (there are fewer individuals with income >$50K). They provide a more robust assessment of the model's ability to correctly identify the minority class.

Precision: 0.7397 | Recall: 0.6199 | F1: 0.6745

## Ethical Considerations

**Bias:** The model is trained on data from 1994 and may perpetuate historical societal biases related to race, sex, and native-country that are present in the data. Applying this model could lead to unfair outcomes for certain demographic groups.

**Outdated Data:** The economic and demographic landscape has changed significantly since 1994. Therefore, the model's learned relationships are not likely to be accurate or relevant for modern populations.

## Caveats and Recommendations

This model should be considered a proof-of-concept for a machine learning pipeline, not a tool for real analysis. It is strongly recommended against using this model for any purpose other than education. If a similar model were to be built for production use, it would require more modern and representative data.