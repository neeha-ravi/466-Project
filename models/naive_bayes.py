from data.data_processing import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder


def run_naive_bayes():
    data = get_data()

    label_encoder = LabelEncoder()
    data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])

    X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = data['Performance Index']

    X_train, X_non_val, y_train, y_non_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(len(X_train))

    X_val, X_test, y_val, y_test = train_test_split(X_non_val, y_non_val, test_size=0.5, random_state=42)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features for NB
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Model
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Validation Mean Squared Error: {val_mse}')

    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print()
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Mean Squared Error: {test_mse}')


    # Predict performance index for a new student with new data
    new_student = pd.DataFrame([[5, 10, 'No', 9, 3]], columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])

    new_student['Extracurricular Activities'] = label_encoder.transform(new_student['Extracurricular Activities'])

    new_student_scaled = scaler.transform(new_student)
    predicted_performance_index = model.predict(new_student_scaled)
    print()
    print(f'Predicted Performance Index for the new student: {predicted_performance_index[0]}')
