
import pandas as pd
from numpy import random
random.seed(seed=12)

# https://github.com/rinikerlab/GHOST/blob/main/notebooks/library_example.ipynb

def read_data():
    dataset = pd.read_csv(r'signalfeat.csv')  # Get the data
    X = dataset.iloc[:, :-1].values  # We got rid of the class value to focus on the FHR diagnosis
    y = dataset.iloc[:, -1].values  # Containing the FHR diagnosis
    ########################################
    # the y-label made a mistake between 1 and 2.
    #########################################
    # Replace 1 with 2 and 2 with 1
    y = [2 if x == 1 else 1 if x == 2 else x for x in y]
    return X, y



