import pickle
import numpy as np

def load_model(file_path='radiology_condition_model.pkl'):
    with open(file_path, 'rb') as f:
        model, label_encoder = pickle.load(f)
    return model, label_encoder

def predict_radiology_condition(image_features, model, label_encoder):
    image_features = np.array(image_features).reshape(1, -1)
    prediction = model.predict(image_features)
    prediction_label = label_encoder.inverse_transform(prediction)
    return prediction_label[0]

if __name__ == "__main__":
    model, label_encoder = load_model()
    # Example synthetic imaging features
    image_features = np.random.rand(100)
    prediction = predict_radiology_condition(image_features, model, label_encoder)
    print(f"Predicted condition for given imaging features: {prediction}")
