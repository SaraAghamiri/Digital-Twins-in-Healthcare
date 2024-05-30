import pickle
from preprocess_data import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    with open('heart_condition_model.pkl', 'wb') as f:
        pickle.dump((model, label_encoder), f)
    print("Model trained and saved to 'heart_condition_model.pkl'")

if __name__ == "__main__":
    train_model()

