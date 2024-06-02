import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

def evaluate_decision_tree():
    data = get_data()

    label_encoder = LabelEncoder()
    data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])

    X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = data['Performance Index']

    X_train, X_non_val, y_train, y_non_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_non_val, y_non_val, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    depth_range = range(1, 21)  # Example range from 1 to 20
    cv_scores = []

    # Perform cross-validation for each depth value
    for depth in depth_range:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(tree, X, y, cv=10, scoring='accuracy')  # 10-fold cross-validation
        cv_scores.append(np.mean(scores))

    # Determine the optimal depth
    optimal_depth = depth_range[np.argmax(cv_scores)]
    print(f"Optimal tree depth: {optimal_depth}")

    model = DecisionTreeClassifier(max_depth=optimal_depth)
    model.fit(X_train, y_train)

    # validation set evaluation
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Mean Squared Error: {val_mse:.4f}')

    # test set evaluation
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Mean Squared Error: {test_mse:.4f}')

    # strat. cv
    skf = StratifiedKFold(n_splits=5)
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=skf)
    print(f'Stratified Cross-Validation Scores: {cross_val_scores}')
    print(f'Stratified Cross-Validation Mean Score: {cross_val_scores.mean():.4f}')

    # learning curve
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 50)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, validation_scores_mean, label='Cross-validation score')
    plt.title('Learning Curve for Decsion Tree Model')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # classification report
    print('Classification Report:')
    print(classification_report(y_test, y_test_pred, zero_division=0))

def get_data():
    file_path = 'data/Student_Performance.csv'
    data = pd.read_csv(file_path)
    return data

if __name__ == "__main__":
    evaluate_decision_tree()
