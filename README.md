# Assessing the Elements Affecting Academic Achievement



This project aims to analyze the factors that contribute to the academic performance of students and aims to answer the important question of what factors have a major impact on GPA. Based on the hypothesis that hours studied, previous scores, extracurricular activities, sleep hours, and sample question papers practiced are important factors in determining performance. Our main objective is to successfully predict performance and provide useful insights by using skills learned in class to establish the connections between these predictor variables and the performance index of a student.

## Requirements

- Python 3.x
- pandas
- scikit-learn

Install the required packages using pip:

```bash
pip3 install pandas scikit-learn
```

## Usage

- Clone the repository
- Run the main script:

```bash
python3 run_models.py
```

## Structure

- **run_models.py**: Main script that runs all fo out models.
- **naive_bayes.py**: Script that runs our naive bayes model.
- **data_processing.py**: Pre-processes our data and holds functions to get any information from it.

- **Student_Performance.csv**: .csv file that holds our data of size 10000

## Workflow

- **Data Preparation**: The dataset is loaded and pre-processed.

- **Model Training**: The Models are trained on the training set.
- **Evaluation**: The model is evaluated on the validation and test sets using accuracy and MSE metrics.
- **Prediction**: The model predicts the performance index for a new student based on the corresponding input data.
