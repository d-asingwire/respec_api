
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load list of files in an activity folder
def load_files_from_folder(folder_path):
    """
    Load all CSV files from a folder and return a list of file paths.

    Parameters:
    folder_path (str): The path to the folder containing CSV files.

    Returns:
    list: A list of file paths for all CSV files in the folder.
    """

    # Initialize an empty list to store the full file paths of the CSV files
    file_paths = []

    # Loop through all the files in the given folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .csv extension (ignores other files)
        if file_name.endswith('.csv'):
            # Construct the full file path by joining the folder path and the file name
            full_file_path = os.path.join(folder_path, file_name)

            # Append the full file path to the file_paths list
            file_paths.append(full_file_path)

    # Return the complete list of CSV file paths
    return file_paths


# Split files

#  Train and test set split from list of files
def split_files(file_list, test_size=0.2):
    """
    Split the list of files into training and test sets.

    Parameters:
    file_list (list): List of file paths to be split into train and test sets.
    test_size (float): The proportion of files to allocate to the test set.
                       Default is 0.2, meaning 20% of the files will be used for testing.

    Returns:
    tuple:
        - train_files (list): List of file paths for the training set.
        - test_files (list): List of file paths for the test set.
    """

    # Split the file list into training and test sets using train_test_split from scikit-learn
    # test_size defines the proportion of the data to use as the test set (default is 20%)
    # shuffle=True ensures that the files are shuffled randomly before splitting
    train_files, test_files = train_test_split(file_list, test_size=test_size, shuffle=True)

    # Return the train and test file lists
    return train_files, test_files


# Sliding window
def load_and_apply_sliding_windows(file_paths, window_size, step_size, label):
    """
    Load accelerometer data from each file, apply sliding windows, and return windows and labels.

    Parameters:
    ----------
    file_paths : list of str
        List of file paths to CSV files containing sensor data (with columns accel_x, accel_y, accel_z).
    window_size : int
        Number of time steps per sliding window.
    step_size : int
        Stride (step) between consecutive windows.
    label : int or str
        Activity label assigned to each window.

    Returns:
    -------
    tuple (windows, labels)
        windows : np.ndarray
            3D array of shape (num_windows, window_size, 3)
            Each window contains accelerometer readings (x, y, z).
        labels : np.ndarray
            1D array of length num_windows containing the activity label for each window.
    """
    windows = []
    labels = []

    for file_path in file_paths:
        # Load only the accelerometer columns
        data = pd.read_csv(file_path, usecols=['accelX', 'accelY', 'accelZ'])
        data = data.to_numpy()
        num_samples = data.shape[0]

        # Apply sliding windows
        for i in range(0, num_samples - window_size + 1, step_size):
            window = data[i:i + window_size]
            windows.append(window)
            labels.append(label)

    return np.array(windows), np.array(labels)


# Load and Split Train Test for Each Activity Folder
def process_activity(activity, label, dataset_path, window_size=50, step_size=50, test_size=0.2):
    """
    Processes an activity folder by loading the file list, splitting them into
    train and test sets, and applying sliding windows to the files.

    Args:
        activity (str): Name of the activity (folder name). This refers to the specific physical activity
                        like 'walking', 'running', etc.
        label (int): Numeric label corresponding to the activity, used for classification.
        dataset_path (str): Base path where the activity folders are located.
        window_size (int): Size of the sliding window, i.e., the number of time steps included in each window.
                           Default is 50.
        step_size (int): Step size for the sliding window, i.e., how far the window moves along the data.
                         Default is 50 (no overlap between windows).
        test_size (float): Proportion of files to use for testing. Default is 0.2, meaning 20% of files will
                           be allocated to the test set.

    Returns:
        tuple:
            - train_windows (numpy.ndarray): Sliding windows from the training files.
            - train_labels (numpy.ndarray): Corresponding labels for the training windows.
            - test_windows (numpy.ndarray): Sliding windows from the test files.
            - test_labels (numpy.ndarray): Corresponding labels for the test windows.
    """
    # Construct the full folder path where the activity files are stored
    folder_path = os.path.join(dataset_path, activity)

    # Load all CSV file paths for the given activity from the folder
    file_list = load_files_from_folder(folder_path)

    # Split the file list into training and testing sets
    # train_files: files used for training
    # test_files: files used for testing
    train_files, test_files = split_files(file_list, test_size=test_size)

    # Apply sliding windows to the training files
    # The function 'load_and_apply_sliding_windows' returns the sliding windows (segments) and their corresponding labels
    train_windows, train_labels = load_and_apply_sliding_windows(train_files, window_size, step_size, label)

    # Apply sliding windows to the testing files
    test_windows, test_labels = load_and_apply_sliding_windows(test_files, window_size, step_size, label)

    # Return the sliding windows and their labels for both training and testing sets
    return train_windows, train_labels, test_windows, test_labels


# Combine Data

def combine_data(train_test_data, data_type):
    """
    Combines the sliding windows and labels from all activities into a single
    array for either training or testing.

    Args:
        train_test_data (dict): Dictionary containing the sliding window data for all activities.
                                Each key in the dictionary corresponds to an activity, and the value is another
                                dictionary with the keys 'train_windows', 'train_labels', 'test_windows', 'test_labels'.
        data_type (str): Either 'train' or 'test' to specify which data to combine (e.g., 'train_windows' or 'test_windows').

    Returns:
        tuple:
            - windows (numpy.ndarray): Concatenated windows from all activities for either training or testing.
            - labels (numpy.ndarray): Concatenated labels corresponding to the windows from all activities.
    """

    # Extract the list of sliding windows for the specified data type (either 'train' or 'test') from each activity
    # For example, if data_type is 'train', it extracts 'train_windows' for all activities
    windows_list = [train_test_data[activity][f'{data_type}_windows'] for activity in train_test_data]

    # Similarly, extract the list of labels corresponding to the windows for each activity
    labels_list = [train_test_data[activity][f'{data_type}_labels'] for activity in train_test_data]

    # Concatenate all the sliding windows into a single numpy array along the first axis (rows)
    # This creates one large array of windows from all the activities combined
    concatenated_windows = np.concatenate(windows_list, axis=0)

    # Concatenate all the labels into a single numpy array along the first axis (rows)
    # The labels are now aligned with the concatenated windows
    concatenated_labels = np.concatenate(labels_list, axis=0)

    # Return the concatenated windows and labels as a tuple
    return concatenated_windows, concatenated_labels


# 1 CNN Model

def build_1d_cnn_model(input_shape, num_classes):
    """
    Builds and compiles a 1D CNN model for multi-class classification.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).
        num_classes (int): The number of output classes.

    Returns:
        model (Sequential): Compiled 1D CNN model.
    """
    model = Sequential()

    # First Conv1D layer
    # You can try experimenting with different filters, kernel_size values and activiation functions
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Second Conv1D layer
    # You can try experimenting with different filters, kernel_size values and activiation functions
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the output from the convolutional layers
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout layer for regularization
    # You can try experimenting with different dropout rates
    model.add(Dropout(0.5))

    # Output layer with softmax for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #  Prints a detailed summary of the model, showing the layers, their output shapes, and the number of trainable parameters
    model.summary()

    return model