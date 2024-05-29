from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

app = Flask(__name__)

#Importar modelo treinado
directory = 'C:/cacau/modelos/'
model = tf.keras.models.load_model(directory + 'model_w.h5')

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_image_from_base64(base64_data):
    try:
        # Decodifica a string base64 em bytes
        img_data = base64.b64decode(base64_data)

        # Converte os bytes para um objeto StringIO
        img_io = io.BytesIO(img_data)

        # Abre a imagem usando o PIL
        img = Image.open(img_io)

        # Converte a imagem para o formato JPEG (opcional, dependendo do formato de entrada)
        img = img.convert('RGB')

        return img

    except Exception as e:
        raise ValueError(f"Erro ao processar dados base64: {str(e)}")

@app.route('/')
@app.route('/index')
def index():
    return "API em Flask está funcionando!"

# Rota de Previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_base64 = request.get_data(as_text=True)
        print('String base64:', image_base64)

        img = get_image_from_base64(image_base64)
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)

        # Aqui você pode processar a saída do modelo conforme necessário
        # Por exemplo, converter para uma resposta JSON
        result = {'prediction': prediction.tolist()}
        print(result)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Erro ao processar dados base64: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug='true')
