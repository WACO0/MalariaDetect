from flask import Flask, render_template, request, make_response
from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Crear una aplicación Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Cargar el modelo
model = keras.models.load_model('modeloMalaria.h5')

# Ruta principal para cargar la página web
@app.route('/')
def index():
    return render_template('index.html')
# Rutas de las paginas
@app.route('/malariaL')
def malariaL():
    return render_template('malariaL.html')

@app.route('/malariaH')
def malariaH():
    return render_template('malariaH.html')



# Ruta para manejar las solicitudes de predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen enviada desde la página web
    image = request.files['image']
    print(image)
    # Preprocesar la imagen
    # Aquí debes implementar el código necesario para leer y preprocesar la imagen según tus necesidades
    # Por ejemplo, puedes utilizar bibliotecas como OpenCV o PIL para cargar y procesar la imagen
    # Asegúrate de redimensionar la imagen y aplicar cualquier otra transformación requerida por tu modelo

    # Realizar la predicción 
    image = Image.open(image).convert('RGB')
    image = image.resize((64, 64))
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0
    prediction = model.predict(img_array)

    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)

    # Devolver la respuesta en formato JSON
    response = {'predicted_class': int(predicted_class)}  # Convertir a int

    # Convertir el diccionario en una cadena JSON
    # Convertir predicted_class a int
    predicted_class = predicted_class.item()
    
    if predicted_class == 0:
        response_text = 'Usted tiene malaria'
    else:
        response_text = 'Usted no tiene malaria'

    # Crear una respuesta JSON
    response = {'predicted_class': response_text}

    # Convertir el diccionario en una cadena JSON
    json_response = json.dumps(response)

    # Crear una respuesta HTTP con la cadena JSON
    http_response = make_response(json_response)

    # Establecer el encabezado Content-Type a application/json
    http_response.headers['Content-Type'] = 'application/json'

    return http_response

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True)