import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    image_features = np.random.rand(num_samples, 100)  # 100 synthetic imaging features
    diagnoses = np.random.choice(['Normal', 'Pneumonia', 'Fracture'], size=num_samples, p=[0.7, 0.2, 0.1])
    data = pd.DataFrame(image_features, columns=[f'Feature_{i}' for i in range(100)])
    data['Diagnosis'] = diagnoses
    return data

if __name__ == "__main__":
    data = generate_synthetic_data()
    data.to_csv('synthetic_radiology_data.csv', index=False)
    print("Synthetic radiology data generated and saved to 'synthetic_radiology_data.csv'")
