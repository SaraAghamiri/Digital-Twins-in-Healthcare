import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    heart_rates = np.random.randint(60, 100, size=num_samples)
    diagnoses = np.random.choice(['Normal', 'Arrhythmia', 'Tachycardia'], size=num_samples, p=[0.7, 0.2, 0.1])
    data = pd.DataFrame({'HeartRate': heart_rates, 'Diagnosis': diagnoses})
    return data

if __name__ == "__main__":
    data = generate_synthetic_data()
    data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved to 'synthetic_data.csv'")
