import pandas as pd

def get_data():
    file_path = 'data/Student_Performance.csv'  
    data = pd.read_csv(file_path)
    return data

def get_rows(n):
    data = get_data()
    return data.head(n)

def get_summary():
    data  = get_data()
    print("\nSummary statistics:")
    print(data.describe())

def get_missing_values():
    data = get_data()
    null_data = data.isnull().sum()
    if null_data == 0:
        return None
    return null_data


# correlation_matrix = data.corr()
# print("\nCorrelation matrix:")
# print(correlation_matrix)
