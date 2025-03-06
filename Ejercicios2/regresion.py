import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #para graficar
from sklearn.linear_model import LinearRegression
import io 
import base64

#data
data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}
#monta el modelo
df = pd.DataFrame(data)
x= df[['Study Hours']]
y= df[['Final Grade']]

#Entrena el modelo
model = LinearRegression()
model.fit(x,y)
# Predice la nota segun las horas de estudio
def calculateGrade(hours):
    result = model.predict([[hours]])[0][0]

    # Generar la gráfica con la predicción
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Datos reales")
    ax.plot(x, model.predict(x), color='red', label="Regresión lineal")
    ax.scatter([[hours]], [[result]], color='green', label="Predicción")  # Punto de predicción
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Final Grade')
    ax.set_title('Final Grade vs Study Hours')
    ax.legend()

    # Convertir la imagen en base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return result, image_base64

#Función para generar la imagen de la gráfica en base64
def generate_plot():
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Datos reales")
    ax.plot(x, model.predict(x), color='red', label="Regresión lineal")
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Final Grade')
    ax.set_title('Final Grade vs Study Hours')

    # Guardar la imagen en un buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    plt.close(fig)

    # Convertir la imagen en base64
    return image_base64