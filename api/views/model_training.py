import pickle
import os
import pandas as pd
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class TrainModel(APIView):

    def post(self, request):
        try:
            # Load dataset
            csv_path = os.path.join(settings.BASE_DIR, 'api', 'datasets', 'house_prices.csv')
            data = pd.read_csv(csv_path)

            # Prepare the data
            X = data[['area', 'bedrooms']]
            y = data['price']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Save the model
            model_path = os.path.join(settings.BASE_DIR, 'api', 'datasets', 'model.pkl')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            return Response(
                {"message": "Model trained and saved successfully", "model_path": model_path},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)