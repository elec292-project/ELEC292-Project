import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
model_filename = '../outputs/model.joblib'
model = joblib.load(model_filename)

# Constants
EXPECTED_FEATURES = 2290


# Function to load CSV file
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        # Load CSV file as numpy array
        data = pd.read_csv(file_path).values

        # Split into 5-second windows based on time differences
        start_time = data[0, 0]
        chunks = []
        current_chunk = []

        for row in data:
            time = row[0]
            if time - start_time < 5:
                current_chunk.append(row)
            else:
                if len(current_chunk) > 0:
                    chunks.append(np.array(current_chunk))
                current_chunk = [row]
                start_time = time

        # Add last chunk (if it exists)
        if len(current_chunk) > 0:
            chunks.append(np.array(current_chunk))

        # Process chunks into model-friendly format
        processed_chunks = []
        for chunk in chunks:
            # Flatten the chunk
            flattened_chunk = chunk.flatten()

            if flattened_chunk.shape[0] < EXPECTED_FEATURES:
                # Pad with zeros if too small
                padding = np.zeros(EXPECTED_FEATURES - flattened_chunk.shape[0])
                flattened_chunk = np.concatenate([flattened_chunk, padding])
            elif flattened_chunk.shape[0] > EXPECTED_FEATURES:
                # Trim if too large
                flattened_chunk = flattened_chunk[:EXPECTED_FEATURES]

            processed_chunks.append(flattened_chunk)

        processed_chunks = np.array(processed_chunks)

        # Predict using the trained model
        predictions = model.predict(processed_chunks)
        labels = ["walking" if pred == 0 else "jumping" for pred in predictions]

        # Save to output CSV
        output_file = file_path.replace(".csv", "_predictions.csv")
        output_df = pd.DataFrame(labels, columns=["Predicted Activity"])
        output_df.to_csv(output_file, index=False)

        # Plot results
        plot_results(predictions)

        messagebox.showinfo("Success", f"Predictions saved to {output_file}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {e}")


# Function to plot results
def plot_results(predictions):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create color map based on predictions
    colors = ['blue' if pred == 0 else 'red' for pred in predictions]

    # Plot line segments based on predictions
    x = np.arange(len(predictions))
    for i in range(len(predictions) - 1):
        ax.plot(x[i:i + 2], [predictions[i]] * 2, color=colors[i], linewidth=3)

    ax.set_title("Predicted Activities Over Time")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Activity (0 = walking, 1 = jumping)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Walking", "Jumping"])

    # Embed plot into Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Create GUI window
window = tk.Tk()
window.title("Activity Predictor")
window.geometry("600x400")

# Add buttons
load_button = tk.Button(window, text="Load CSV", command=load_csv, padx=10, pady=5)
load_button.pack(pady=20)

# Start GUI event loop
window.mainloop()
