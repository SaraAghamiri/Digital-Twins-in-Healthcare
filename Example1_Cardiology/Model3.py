import os
import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    
    run_script('PatientsData.py')
    
    
    run_script('PatientsData2.py')
    
    
    run_script('TrainModel1.py')
    
    
    run_script('Model2.py')
