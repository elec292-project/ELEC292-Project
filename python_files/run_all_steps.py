import subprocess

# List all the Python files you want to run
python_files = [
    'step2_data_storage.py',
    'step3_visualization_plots.py',
    'step4_preprocessing.py',
    'step5_preprocessed_data_split.py',
    'step5.5_data_features.py',
    'step6_test_and_train.py'
]

# Run each Python file
for file in python_files:
    print(f"Running {file}...")
    result = subprocess.run(['python', file], capture_output=True, text=True)

    # Print the output of each script
    print(f"Output of {file}:")
    print(result.stdout)

    # If there were errors, print them
    if result.stderr:
        print(f"Error in {file}:")
        print(result.stderr)