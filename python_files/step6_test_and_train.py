import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the HDF5 file containing the segmented data
hdf5_path = "../outputs/data.hdf5"

# Open the HDF5 file to access the data
with h5py.File(hdf5_path, "r") as hdf:
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    # Loop through each user in training data
    for user in hdf["segmented/train"]:
        # Loop through each activity in the user's data
        for activity in hdf[f"segmented/train/{user}"]:
            if activity == "walking":
                activity_label = 0
            elif activity == "jumping":
                activity_label = 1

            # Loop through each dataset in the activity
            for dataset_name in hdf[f"segmented/train/{user}/{activity}"]:
                dataset = hdf[f"segmented/train/{user}/{activity}/{dataset_name}"]
                for chunk_name in hdf[f"segmented/train/{user}/{activity}/{dataset_name}"]:
                    chunk = dataset[chunk_name]  # Get chunk
                    data = chunk[:450]  # Take the first 450 rows

                    # Extract statistical attributes from chunk attributes dynamically
                    attribute_values = [chunk.attrs[attr] for attr in chunk.attrs]  # Get all attribute values

                    # Concatenate time-series data with all attributes
                    combined_data = np.concatenate([data.flatten(), attribute_values])

                    train_data.append(combined_data)
                    train_labels.append(activity_label)

    # Loop through each user in testing data
    for user in hdf["segmented/test"]:
        # Loop through each activity in the user's data
        for activity in hdf[f"segmented/test/{user}"]:
            if activity == "walking":
                activity_label = 0
            elif activity == "jumping":
                activity_label = 1

            # Loop through each dataset in the activity
            for dataset_name in hdf[f"segmented/test/{user}/{activity}"]:
                dataset = hdf[f"segmented/test/{user}/{activity}/{dataset_name}"]
                for chunk_name in hdf[f"segmented/test/{user}/{activity}/{dataset_name}"]:
                    chunk = dataset[chunk_name]  # Get chunk
                    data = chunk[:450]  # Take the first 450 rows

                    # Extract statistical attributes from chunk attributes dynamically
                    attribute_values = [chunk.attrs[attr] for attr in chunk.attrs]  # Get all attribute values
                    # Concatenate time-series data with all attributes
                    combined_data = np.concatenate([data.flatten(), attribute_values])

                    test_data.append(combined_data)
                    test_labels.append(activity_label)


pca = PCA(n_components=50)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Define the Logistic Regression model
model = LogisticRegression(max_iter=5000, class_weight='balanced')

# Train the model using the entire training data
model.fit(train_data, train_labels)

# Evaluate on the test set (acting as both validation and test set)
test_predictions = model.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy*100:.4f}")

print("\nBot's Guesses vs Actual Labels:")
for i in range(len(test_predictions)):
    guess = "walking" if test_predictions[i] == 0 else "jumping"
    actual = "walking" if test_labels[i] == 0 else "jumping"
    print(f"Sample {i+1}: Bot's Guess = {guess}, Actual = {actual}")
