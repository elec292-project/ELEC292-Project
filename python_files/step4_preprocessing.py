# Preprocess the data (remove noise)
import h5py
import numpy as np

hdf5_path  = "../outputs/data.hdf5"  # Get HDF5 file

# Some basic filtering
with h5py.File(hdf5_path, 'a') as hdf:
    for user in hdf["raw"]: # Loop through each user
        for activity in hdf[f'raw/{user}']: # Loop through each activity
            for dataset in hdf[f'raw/{user}/{activity}']: # Loop thorugh each dataset
                data = hdf[f'raw/{user}/{activity}/{dataset}'][:] # Get all the data

                data = data[data[:, 0] <= 150] # Filter the data based on if its <=150s

                # if activity == "walking": #If walking
                #     filtered_data = filtered_data[filtered_data[:, 4] < 10] # Filter z acceleration > 10
                #     filtered_data = filtered_data[filtered_data[:, 4] > -10] # Filter z acceleration < 10

                # Moving average filter
                window_size = 10
                for i in range(data.shape[1]):  # Iterate over every column
                    data[:, i] = np.convolve(data[:, i], np.ones(window_size) / window_size, mode='same')

                processed_activity = hdf[f'processed/{user}/{activity}'] # Processed activity group

                if dataset in processed_activity:
                    del processed_activity[dataset] # Delete old processed data if there

                processed_activity.create_dataset(dataset, data=data)

print("Finished Preprocessing Data")