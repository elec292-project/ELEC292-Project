import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew
import matplotlib.pyplot as plt


def preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.to_numpy()

    # filtered_data = data[data[:, 0] <= 150]

    window_size = 10
    for i in range(data.shape[1]):  # Iterate over every column
        data[:, i] = np.convolve(data[:, i], np.ones(window_size) / window_size, mode='same')

    return data


def split_data_into_chunks(filtered_data):
    # Set start and end times, arrays for chunks
    start_time = filtered_data[0, 0]
    chunks = []
    current_chunk = []

    for row in filtered_data:  # Loop through each row
        time = row[0]  # Get the time
        if time - start_time < 5:  # If 5 seconds haven't passed
            current_chunk.append(row)  # Add it to the current chunk
        else:  # If 5 seconds have passed
            if len(current_chunk) > 0:
                chunks.append(np.array(current_chunk))  # Add the chunk to list of all chunks
            current_chunk = [row]  # Set the current row to be the start of the next chunk
            start_time = time  # Set start time to the new current time

    # Final add to list (for last few rows of data)
    if len(current_chunk) > 0:
        chunks.append(np.array(current_chunk))

    return chunks


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
    features = dict(sorted(features.items()))

    feature_values = list(features.values())

    # Convert feature values into a numpy array for z-score standardization
    feature_values = np.array(feature_values)

    # Calculate mean and standard deviation for Z-score standardization
    mean = np.mean(feature_values)
    std_dev = np.std(feature_values)

    # Apply Z-score standardization
    standardized_values = (feature_values - mean) / std_dev

    # Return the list of normalized feature values (without labels)
    return standardized_values



def predict_activity(chunk_features, model):
    predictions = []

    # Loop through each set of chunk features (which are now numpy arrays)
    # Instead of reshaping each feature set, stack all chunk features together into a 2D array
    chunk_features_array = np.array(chunk_features)  # This will have shape (n_chunks, n_features)
    print(chunk_features_array.shape)
    predictions = model.predict(chunk_features_array)  # Directly pass the entire array

    predictions_proba = model.predict_proba(chunk_features_array)
    print("Prediction probabilities:", predictions_proba)

    return predictions




def browse_file():
    # Allow the user to browse for a CSV file
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Preprocess the data
        filtered_data = preprocess_data(file_path)
        # Split the preprocessed data into chunks
        chunks = split_data_into_chunks(filtered_data)

        # Extract features from the chunks
        chunk_features = [normalize_features(extract_features(chunk)) for chunk in chunks]

        # Load the saved model
        model_path = "../outputs/model.pkl"
        model = joblib.load(model_path)

        # Predict the activity for each chunk (Walking or Jumping)
        predictions = predict_activity(chunk_features, model)

        # Prepare the data for output
        output_data = []
        for i, (chunk, prediction) in enumerate(zip(chunks, predictions)):
            # Extract the starting time of each chunk
            start_time = chunk[0, 0]
            activity = "Walking" if prediction == 0 else "Jumping"
            output_data.append([start_time, activity])

        # Convert output data to a DataFrame and save to CSV
        output_df = pd.DataFrame(output_data, columns=["Start Time", "Activity"])
        output_df.to_csv("../outputs/chunk_predictions.csv", index=False)

        messagebox.showinfo("Result", f"Finished processing data. Predictions saved to 'chunk_predictions.csv'.")


def create_gui():
    # Create the main window
    window = tk.Tk()
    window.title("Walking or Jumping?")

    # Set window size
    window.geometry("400x200")

    # Create and pack the widgets
    label = tk.Label(window, text="Select a CSV file to preprocess")
    label.pack(pady=20)

    browse_button = tk.Button(window, text="Browse", command=browse_file)
    browse_button.pack(pady=20)

    # Start the GUI event loop
    window.mainloop()


# Run the GUI application
create_gui()

