# Acceleration vs Time Plots for Dataset
import matplotlib as mpl
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

#######################################################################################################################################################
acceleration_axis = "z"  # Options: x, y, z, abs_acc
output = "../outputs/plots/processed" # Where to store
data_type = "processed" # Data "type" to plot (i.e raw, processed)
#######################################################################################################################################################

hdf5_path  = "../outputs/data.hdf5" # Get HDF5 file

axis_map = {"x": 1, "y": 2, "z": 3, "abs_acc" : 4} # Map axis to the index they are in the HDF5 (it's stored as a numpy)
axis_index = axis_map[acceleration_axis] # Get index

data = {"walking": [], "jumping": []}  # Store data for walking & jumping

# Load all the data
with h5py.File(hdf5_path, "r") as hdf: # Read file
    for user in hdf[data_type]: # Loop through all users
        for activity in ["walking", "jumping"]: # Make sure group is either walking or jumping
            activity_path = f"{data_type}/{user}/{activity}"
            if activity_path in hdf: # If it's in the HDF
                for dataset_name in hdf[activity_path]:  # Loop through datasets
                    dataset = hdf[f"{activity_path}/{dataset_name}"][:]  # Load dataset
                    time = dataset[:, 0]  # Get time
                    acceleration = dataset[:, axis_index]  # Get selected acceleration
                    data[activity].append((time, acceleration))  # Store the data

# Function for plotting data. Passes in LIST of activities, LIST of colors, and filename that it writes to. Must be LISTs (even individual plots need to be in list format)
def plot_activity(activities, colors, filename):
    plt.figure(figsize=(10, 5)) # Create 10x5" figure

    for activity, color in zip(activities, colors): # Loop through each activity, associate it with the color list
        for time, acceleration in data[activity]: # Plot data
            plt.plot(time, acceleration, label=activity.capitalize(), color=color, alpha=0.5)

    plt.xlabel("Time") # Set X Label
    plt.ylabel(f"{acceleration_axis.upper()} Acceleration")  # Set Y Label (dependant on acceleration)

    plt.title(f"{acceleration_axis.upper()} Acceleration for {', '.join([a.capitalize() for a in activities])} Across All Users") # Set title

    plt.legend([a.capitalize() for a in activities]) # Set legend
    plt.grid() # Add grid

    plt.savefig(os.path.join(output, filename), dpi=300) # Save file
    plt.close()

plot_activity(["walking"], ["g"], f"{acceleration_axis}_acceleration_walking.png") # Walking
plot_activity(["jumping"], ["b"], f"{acceleration_axis}_acceleration_jumping.png") # Jumping
plot_activity(["walking", "jumping"], ["g", "b"], f"{acceleration_axis}_acceleration_walking_jumping.png") # Both

