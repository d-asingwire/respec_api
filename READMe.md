# House Price Prediction API

A simple **Django REST API** that trains a **linear regression model** and predicts house prices based on input features such as **square footage (area)** and **number of bedrooms**.

---

## Features

* Train a regression model on a local dataset (`house_prices.csv`).
* Predict house prices given input features.
* RESTful API built with **Django REST Framework**.
* Clean and modular project structure for easy extension.

---

## How It Works

1. The `/api/model/train/` endpoint loads a dataset, trains a `LinearRegression` model, and saves it as `model.pkl`.
2. The `/api/model/predict/` endpoint loads the trained model and returns a predicted price for given inputs (e.g., `area`, `bedrooms`).

---

## Project Structure

```
respec_api/
├── manage.py
├── api/
│   ├── datasets/
│   │   └── house_prices.csv
│   ├── views/
│   │   ├── model_training.py
│   │   └── model_prediction.py
│   └── ...
├── respec_api/
│   ├── settings.py
│   └── urls.py
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/d-asingwire/house-price-api.git
cd respec_api
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Database Migrations

```bash
python manage.py migrate
```

### 5. Start the Development Server

```bash
python manage.py runserver
```

---

## API Endpoints

### 🔹 Train Model

**POST** `/api/model/train/`

Trains the linear regression model using `house_prices.csv` and saves it locally.

**Response Example:**

```json
{
  "message": "Model trained and saved successfully",
  "model_path": "/api/datasets/model.pkl"
}
```

---

### Predict House Price

**POST** `/api/model/predict/`

**Request Body:**

```json
{
  "area": 2100,
  "bedrooms": 3
}
```

**Response Example:**

```json
{
    "predicted_price": "3,493,872.83"
}
```

---

## Sample Dataset Format (`house_prices.csv`)

| area | bedrooms | price  |
| ---- | -------- | ------ |
| 2000 | 3        | 400000 |
| 2500 | 4        | 500000 |
| 1800 | 2        | 350000 |

---

## Technologies Used

* **Python 3.10+**
* **Django 5+**
* **Django REST Framework**
* **pandas**, **scikit-learn**, **pickle**

---

## Author

**Dallington Asingwire**
[LinkedIn](https://www.linkedin.com/in/dallington-asingwire-0468a414a/) | [GitHub](https://github.com/d-asingwire)

---