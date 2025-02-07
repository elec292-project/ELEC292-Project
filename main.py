import pandas as pd
import h5py
import numpy as np

csv_file = ['sindu_jumping_hand.csv', 'sindu_walking_hand.csv']  # File paths
data = pd.read_csv(csv_file)