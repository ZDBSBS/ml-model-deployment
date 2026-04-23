# Model Card

## Model Details
The model is a RandomForestClassifier trained to predict whether an individual earns more than 50K per year based on census data.
The model was trained using scikit-learn.

## Intended Use
This model is intended for educational purposes as part of a machine learning deployment project.
It should not be used for real-world decision-making.

## Training Data
The training data consists of census data from the UCI Machine Learning Repository.
The dataset includes demographic and employment-related attributes such as age, education, workclass, and occupation.

## Evaluation Data
The evaluation data is a held-out test set created using an 80/20 train-test split.

## Metrics
The model is evaluated using precision, recall, and F1 score.
These metrics are suitable for binary classification tasks and help evaluate model performance on imbalanced datasets.

## Quantitative Analysis

On the held-out test dataset, the model achieved the following performance:
- Precision: 0.742
- Recall: 0.638
- F1 Score: 0.686

Performance varies across different slices of the data, as documented in `slice_output.txt`.

## Ethical Considerations
The model may reflect biases present in the training data.
Predictions should be interpreted with caution, especially for underrepresented demographic groups.

## Caveats and Recommendations
The model was trained on a static dataset and may not generalize well to future or different populations.
Further evaluation and bias analysis are recommended before any real-world usage.