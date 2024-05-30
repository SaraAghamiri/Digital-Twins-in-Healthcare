import pickle
import numpy as np

def load_model(file_path='heart_condition_model.pkl'):
    with open(file_path, 'rb') as f:
        model, label_encoder = pickle.load(f)
    return model, label_encoder

def predict_heart_condition(heart_rate, model, label_encoder):
    heart_rate = np.array(heart_rate).reshape(-1, 1)
    prediction = model.predict(heart_rate)
    prediction_label = label_encoder.inverse_transform(prediction)
    return prediction_label[0]

if __name__ == "__main__":
    model, label_encoder = load_model()
    heart_rate = 75  # Example heart rate
    prediction = predict_heart_condition(heart_rate, model, label_encoder)
    print(f"Predicted condition for heart rate {heart_rate}: {prediction}")
