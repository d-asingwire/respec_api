from django.conf import settings
from rest_framework.views import APIView
import numpy as np
import glob
import os

from rest_framework.response import Response
from rest_framework import status
from api.common.preprocessor import combine_data, process_activity, build_1d_cnn_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report

class RESpeckModelTrain(APIView):

    def prepare_data(self, dataset_path):
        activity_paths = glob.glob(os.path.join(dataset_path, "*"))
        # Extract just the folder names (activity names)
        activity_names = [path.split('/')[-1] for path in activity_paths]

        # Create a dictionary mapping activity → numeric label
        activities = {activity: i for i, activity in enumerate(activity_names)}
        print(activities)
        return activities

    def pre_process(self, dataset_path, activities):
        # Dictionary to store sliding windows and labels for both train and test sets for each activity
        # This will hold the training and test data after processing each activity.
        train_test_data = {}

        # Loop through each activity folder and process the data
        # Note, if you have large amounts of data, this step may take a while
        for activity, label in activities.items():
            # Initialize an empty dictionary for each activity to store train and test windows and labels
            train_test_data[activity] = {}

            # Call process_activity() to process the data for the current activity folder
            # It loads the data, applies sliding windows, splits it into train and test sets,
            # and returns the respective sliding windows and labels for both sets.
            (train_test_data[activity]['train_windows'], train_test_data[activity]['train_labels'],
             train_test_data[activity]['test_windows'], train_test_data[activity]['test_labels']) = process_activity(
                activity, label, dataset_path)

        # Explanation:
        # - 'train_windows' and 'train_labels' store the windows and labels from the training files.
        # - 'test_windows' and 'test_labels' store the windows and labels from the test files.
        # - `your_dataset_path` should be replaced with the actual path to your dataset.
        # - `process_activity` handles all the steps of loading data, splitting it, and applying sliding windows.
        print(train_test_data)
        return train_test_data

    def build_model(self, X_train, y_train_one_hot,X_test, y_test_one_hot):
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train_one_hot.shape[1]
        model = build_1d_cnn_model(input_shape, num_classes)
        return model


    def train_model(self, model, X_train, y_train_one_hot, X_test, y_test_one_hot):
        history = model.fit(X_train, y_train_one_hot,
                            epochs=20,  # Train the model for 20 epochs
                            batch_size=32,  # Use a batch size of 32
                            validation_data=(X_test, y_test_one_hot))  # Validate on the test set after each epoch
        return history


        
    def evaluate_model(self, model, X_test, y_test_one_hot):
        # Get predicted probabilities for the test set
        y_pred_probs = model.predict(X_test)
        # Convert the predicted probabilities to class labels (taking the argmax of the probabilities)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        # Convert the true test labels from one-hot encoding back to class labels
        y_true_classes = np.argmax(y_test_one_hot, axis=1)
        # Generate the classification report
        report = classification_report(y_true_classes, y_pred_classes, digits=4)
        # Print the classification report
        print(report)

    def export_model(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)  # model is your trained Keras model
        tflite_model = converter.convert()

        # Save the converted model to a .tflite file
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

        return "Model successfully exported to model.tflite"

    def post(self, request):
        try:
            dataset_path = os.path.join(settings.BASE_DIR, 'api', 'content', 'RESpeckData', 'daily_activity')
            data = {dataset_path}
            glob.glob(os.path.join(dataset_path, "*"))

            activity_folder_name = "ascending"
            pattern = os.path.join(dataset_path, activity_folder_name, "*")
            files = glob.glob(pattern)

            # Print first 10 files
           # print(files[:10])

            # Build Classification Pipeline
            activities = self.prepare_data(dataset_path)
            train_test_data = self.pre_process(dataset_path, activities)

            X_train, y_train = combine_data(train_test_data, 'train')
            X_test, y_test = combine_data(train_test_data, 'test')
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            encoder = OneHotEncoder(sparse_output=False)
            y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
            y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
            print(f"y_train_one_hot shape: {y_train_one_hot.shape}, y_test_one_hot shape: {y_test_one_hot.shape}")

            # build model
            model = self.build_model(X_train,y_train_one_hot, X_test, y_test_one_hot)
            # train model
            self.train_model(model, X_train, y_train_one_hot, X_test, y_test_one_hot)
            self.evaluate_model(model, X_test, y_test_one_hot)
            msg = self.export_model(model)

            return Response(
                {"message": msg, "data": data},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)