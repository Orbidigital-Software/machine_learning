import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "files", "Consumo de combustible.xlsx")

# Cargar datos
df = pd.read_excel(FILE_PATH, sheet_name="Sheet1")

# Variables
X = df[["Velocidad (km/h)"]]
y = df["Consumo (L/100km)"]

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación del modelo
def evaluate_model():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, model.coef_[0], model.intercept_

# Función para predecir consumo
def predict_consumption(speed):
    prediction = model.predict([[speed]])[0]
    return prediction

# Función para generar gráfica y convertirla a Base64
def generate_plot():
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test["Velocidad (km/h)"], y=y_test, label="Datos reales")
    sns.lineplot(x=X_test["Velocidad (km/h)"], y=y_pred, color="red", label="Regresión Lineal")
    plt.xlabel("Velocidad (km/h)")
    plt.ylabel("Consumo (L/100km)")
    plt.title("Regresión Lineal: Velocidad vs Consumo de Combustible")
    plt.legend()

    # Guardar la imagen en un buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    return image_base64