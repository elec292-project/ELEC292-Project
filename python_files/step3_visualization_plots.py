# Acceleration vs Time Plots for Dataset
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

#######################################################################################################################################################
acceleration_axes = ["x", "y", "z", "abs_acc"]  # Options: x, y, z, abs_acc
output = "../outputs/plots/raw" # Where to store
data_type = "raw" # Data "type" to plot (i.e raw, processed)
#######################################################################################################################################################

hdf5_path = "../outputs/data.hdf5" # Get HDF5 file

axis_map = {"x": 1, "y": 2, "z": 3, "abs_acc": 4} # Map axis to the index they are in the HDF5 (it's stored as a numpy)
axis_indices = {axis: axis_map[axis] for axis in acceleration_axes} # Get indices for all axes

# Store data for walking & jumping per axis
data = {axis: {"walking": [], "jumping": []} for axis in acceleration_axes}

# Load all the data
with h5py.File(hdf5_path, "r") as hdf: # Read file
    for user in hdf[data_type]: # Loop through all users
        for activity in ["walking", "jumping"]: # Make sure group is either walking or jumping
            activity_path = f"{data_type}/{user}/{activity}"
            if activity_path in hdf: # If it's in the HDF
                for dataset_name in hdf[activity_path]:  # Loop through datasets
                    dataset = hdf[f"{activity_path}/{dataset_name}"][:]  # Load dataset
                    time = dataset[:, 0]  # Get time
                    for axis, idx in axis_indices.items():
                        acceleration = dataset[:, idx]  # Get selected acceleration
                        data[axis][activity].append((time, acceleration))  # Store the data

# Function for plotting data across multiple axes in one file
def plot_activity(activities, colors, axes, filename_prefix):
    fig, axs = plt.subplots(len(axes), 1, figsize=(12, 4 * len(axes)), sharex=True) # Create subplot for each axis

    for i, axis in enumerate(axes): # Loop through each axis
        for activity, color in zip(activities, colors): # Loop through each activity, associate it with the color list
            for time, acceleration in data[axis][activity]: # Plot data
                axs[i].plot(time, acceleration, label=f"{activity.capitalize()}", color=color, alpha=0.5)

        axs[i].set_ylabel(f"{axis.upper()} Acceleration")  # Set Y Label
        axs[i].set_title(f"{axis.upper()} Acceleration for {', '.join([a.capitalize() for a in activities])}") # Set title
        axs[i].legend() # Set legend
        axs[i].grid(True) # Add grid

    axs[-1].set_xlabel("Time") # Only set X label on the last subplot

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save file with consistent naming format
    filename = f"{'_'.join(axes)}_{filename_prefix}.png"
    plt.savefig(os.path.join(output, filename), dpi=300)
    plt.close()

# Plot all activities on multiple axes
plot_activity(["walking"], ["g"], acceleration_axes, f"{acceleration_axes}_acceleration_walking.png") # Walking
plot_activity(["jumping"], ["b"], acceleration_axes, f"{acceleration_axes}_acceleration_jumping.png") # Jumping
plot_activity(["walking", "jumping"], ["g", "b"], acceleration_axes, f"{acceleration_axes}_acceleration_walking_jumping.png") # Both

print("Finished Creating Plots")
