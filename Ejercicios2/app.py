import re
from flask import Flask, render_template, request
from datetime import datetime
import regresion

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Class!'

@app.route('/hello/<name>')
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z]+", name)
    if match_object : 
        clear_name = match_object.group(0)
    else: 
        clear_name = "Friend"

    content = f"hello there,   {clear_name}  ! Hour: {now}"
    return content

@app.route('/exampleMLHTML/')
def exampleHTML():
    return render_template('example.html')

@app.route('/linearRegression/', methods=['GET', 'POST'])
def linearRegression():
    calculateResult  = None
    image_base64 = regresion.generate_plot() #Genera la gráfica
    print("Generated image (base64): ", image_base64[:100])
    if request.method == 'POST':
        hours = float(request.form['hours'])
        calculateResult, image_base64 = regresion.calculateGrade(hours)
    return render_template("linearRegression.html", result = calculateResult, image = image_base64)

if __name__ == '__main__':
    app.run(debug=True)