import os
import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    # Step 1: Generate Synthetic Data
    run_script('PatientsData.py')
    
    # Step 2: Preprocess Data
    run_script('preprocess_data.py')
    
    # Step 3: Train AI Model
    run_script('train_model.py')
    
    # Step 4: Predict with the AI Model
    run_script('predict.py')
