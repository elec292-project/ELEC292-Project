import h5py
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Load the HDF5 file containing the segmented data
hdf5_path = "../outputs/data.hdf5"

# Define feature extraction function
def extract_features(chunk):
    features = {}

    # Ignore the first column (time)
    chunk_data = chunk[:, 1:]  # Select all rows and columns except the first

    # Define the column names
    column_names = ["x", "y", "z", "abs_acc"]

    # Loop over each column in the chunk (excluding the time column)
    for col_idx in range(chunk_data.shape[1]):
        column = chunk_data[:, col_idx]  # Get the data for the current column
        col_name = column_names[col_idx]  # Get the column name

        # Calculate various statistical features for the current column
        features[f'{col_name}_mean'] = np.mean(column)
        features[f'{col_name}_median'] = np.median(column)
        features[f'{col_name}_min'] = np.min(column)
        features[f'{col_name}_max'] = np.max(column)
        features[f'{col_name}_range'] = np.ptp(column)  # Range (max - min)
        features[f'{col_name}_variance'] = np.var(column)
        features[f'{col_name}_std_dev'] = np.std(column)
        features[f'{col_name}_skewness'] = skew(column)
        features[f'{col_name}_kurtosis'] = np.max(column) - np.min(column)  # Rough kurtosis measure
        features[f'{col_name}_sum'] = np.sum(column)

    return features

def normalize_features(features):
    # Extract feature values as a list
    feature_values = list(features.values())

    # Convert feature values into a numpy array for z-score standardization
    feature_values = np.array(feature_values)

    # Calculate mean and standard deviation for Z-score standardization
    mean = np.mean(feature_values)
    std_dev = np.std(feature_values)

    # Apply Z-score standardization
    standardized_values = (feature_values - mean) / std_dev

    # Return normalized features as a dictionary with the original feature names
    return dict(zip(features.keys(), standardized_values))

# Open the HDF5 file to access the data
with h5py.File(hdf5_path, "a") as hdf:
    # Loop through each user in training data
    for user in hdf["segmented/train"]:
        # Loop through each activity in the user's data
        for activity in hdf[f"segmented/train/{user}"]:
            # Loop through each dataset in the activity
            for dataset_name in hdf[f"segmented/train/{user}/{activity}"]:
                dataset = hdf[f"segmented/train/{user}/{activity}/{dataset_name}"]
                for chunk_name in hdf[f"segmented/train/{user}/{activity}/{dataset_name}"]:
                    chunk = dataset[chunk_name] # Get chunk
                    features = extract_features(chunk)
                    normalized_features = normalize_features(features)

                    # Add the normalized features as attributes to the chunk
                    chunk.attrs.update(normalized_features)

    # Loop through each user in testing data
    for user in hdf["segmented/test"]:
        # Loop through each activity in the user's data
        for activity in hdf[f"segmented/test/{user}"]:
            # Loop through each dataset in the activity
            for dataset_name in hdf[f"segmented/test/{user}/{activity}"]:
                dataset = hdf[f"segmented/test/{user}/{activity}/{dataset_name}"]
                for chunk_name in hdf[f"segmented/test/{user}/{activity}/{dataset_name}"]:
                    chunk = dataset[chunk_name] # Get chunk
                    features = extract_features(chunk)
                    normalized_features = normalize_features(features)

                    # Add the normalized features as attributes to the chunk
                    chunk.attrs.update(normalized_features)

print("Finished Extracting Features")
