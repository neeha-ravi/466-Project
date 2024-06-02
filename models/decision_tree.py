from data.data_processing import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder


def run_decision_tree(values):
    data = get_data()

    label_encoder = LabelEncoder()
    data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])

    X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = data['Performance Index']

    X_train, X_non_val, y_train, y_non_val = train_test_split(X, y, test_size=0.2, random_state=30)
    X_val, X_test, y_val, y_test = train_test_split(X_non_val, y_non_val, test_size=0.5, random_state=30)

    # Model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Validation Mean Squared Error: {val_mse}')

    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f'\nTest Accuracy: {test_accuracy}')
    print(f'Test Mean Squared Error: {test_mse}')


    # Predict performance index for a new student with new data
    new_student = pd.DataFrame([values], columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])
    new_student['Extracurricular Activities'] = label_encoder.transform(new_student['Extracurricular Activities'])

    predicted_performance_index = model.predict(new_student)
    print(f'\nPredicted Performance Index for the new student: {predicted_performance_index[0]}')

    metrics = f"""
Validation Accuracy: {val_accuracy}
Validation Mean Squared Error: {val_mse}
Test Accuracy: {test_accuracy}
Test Mean Squared Error: {test_mse}
    """

    return [predicted_performance_index[0], metrics]
