# Making HDF5 File
import h5py
import pandas as pd
import numpy as np
import os
from glob import glob

hdf5_path = "../outputs/data.hdf5"  # Output Path

# Raw Data Folders
data_folders = {
    "sindu": "../raw_data/sindu",
    "logan": "../raw_data/logan",
    # "finn": "../raw_data/finn",
}

# Required Data Columns, assigned to keywords
required_columns = {
    "Time (s)": "time",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Absolute acceleration (m/s^2)": "abs_acc"
}

with h5py.File(hdf5_path, "w") as hdf: # Opening/Creating HDF5 File
    # Raw Data & Pre-Processed Data (Need User Groups!)
    for user, folder in data_folders.items(): # Iterate over each user/folder
        user_raw_group = hdf.create_group(f"raw/{user}")  # Create user group based of user
        user_processed_group = hdf.create_group(f"processed/{user}")  # Create user group based of user

        user_walking_data_group = hdf.create_group(f"raw/{user}/walking")  # Create another hdf group
        user_jumping_data_group = hdf.create_group(f"raw/{user}/jumping")  # Create another hdf group
        user_processed_walking_group = hdf.create_group(f"processed/{user}/walking")
        user_processed_jumping_group = hdf.create_group(f"processed/{user}/jumping")

        csv_files = glob(os.path.join(folder, "*.csv")) # Get all files in directory

        for file in csv_files:
            df = pd.read_csv(file) # Read CSV into DataFrame

            if set(required_columns.keys()).issubset(df.columns): # Check to make sure all columns are there (should always be true due to how data is collected)
                df = df.rename(columns = required_columns) # Rename columns to easier keywords
                data_array = df[['time', 'x', 'y', 'z', 'abs_acc']].to_numpy() # Convert to numpy array

                dataset_name = os.path.splitext(os.path.basename(file))[0] # Gets name of dataset without .csv (for naming)
                dataset_type = dataset_name.split("_")[1] # Get dataset type (always second word, i.e sindu_walking_etc)

                if dataset_type == "walking": # If walking
                    user_walking_data_group.create_dataset(dataset_name, data=data_array)  # Store into HDF5
                elif dataset_type == "jumping": # If jumping
                    user_jumping_data_group.create_dataset(dataset_name, data=data_array)  # Store into HDF5



    # Segmented Data
    train_group = hdf.create_group("segmented/train") # Train Group
    test_group = hdf.create_group("segmented/test") # Test Group

print("Finished Data Storage")


