import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictPrice(APIView):
    def post(self, request):
        try:
            data = request.data
            # Load the model
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)

            # Prepare the input data
            sqft = data.get('area')
            bedrooms = data.get('bedrooms')
            prediction = model.predict([[sqft, bedrooms]])

            return Response({"predicted_price": "{:,.2f}".format(prediction[0])})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)