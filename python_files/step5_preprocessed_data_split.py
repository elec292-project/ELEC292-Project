import h5py
import numpy as np

hdf5_path = "../outputs/data.hdf5"

with h5py.File(hdf5_path, "a") as hdf:
    for user in hdf["processed"]: # Loop through users
        for activity in hdf[f"processed/{user}"]: # Loop through actvities
            dataset = hdf[f"processed/{user}/{activity}"] # Get dataset

            for dataset_name in dataset:
                data = dataset[dataset_name][()]  # Load data as numpy array

                # Set start and end times, arrays for chunks
                start_time = data[0, 0]
                chunks = []
                current_chunk = []

                for row in data: # Loop through each row
                    time = row[0] # Get the time
                    if time - start_time < 5: # If 5 seconds havent passed
                        current_chunk.append(row) # Add it to the current chunk
                    else: # If 5 seconds has passed
                        if len(current_chunk) > 0:
                            chunks.append(np.array(current_chunk)) # Add the chunk to list of all chunks
                        current_chunk = [row] # Set the current row to be the start of the next chunk
                        start_time = time # Set start time to the new current time

                # Final add to list (for last few rows of data)
                if len(current_chunk) > 0:
                    chunks.append(np.array(current_chunk))

                # Shuffle the chunks
                np.random.shuffle(chunks)

                # Split into 90% train and 10% test
                split_index = int(len(chunks) * 0.9)
                train_chunks = chunks[:split_index]
                test_chunks = chunks[split_index:]

                # Create train group
                if f"segmented/train/{user}/{activity}/{dataset_name}" in hdf:
                    del hdf[f"segmented/train/{user}/{activity}/{dataset_name}"] # If existing already, delete
                train_group = hdf.create_group(f"segmented/train/{user}/{activity}/{dataset_name}")

                # Create test group
                if f"segmented/test/{user}/{activity}/{dataset_name}" in hdf:
                    del hdf[f"segmented/test/{user}/{activity}/{dataset_name}"] # If existin alraedy, delete
                test_group = hdf.create_group(f"segmented/test/{user}/{activity}/{dataset_name}")

                for i, chunk in enumerate(train_chunks): # Add each chunk to train group
                    train_group.create_dataset(f"chunk_{i}", data=chunk)

                for i, chunk in enumerate(test_chunks): # Add each chunk to test group
                    test_group.create_dataset(f"chunk_{i}", data=chunk)

print("Finished Splitting Data into Chunks")
