import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path='synthetic_data.csv'):
    data = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])
    X = data[['HeartRate']]
    y = data['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data()
    print("Data preprocessed and split into training and testing sets.")
