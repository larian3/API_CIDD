from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from pathlib import Path

app = Flask(__name__)


model = tf.keras.models.load_model('../model.h5')

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

        # Obtém todas as classes previstas e suas confianças
        predicted_classes = np.argsort(prediction[0])[::-1]  # Obtém as classes ordenadas por confiança
        prediction_results = {}
        for i, class_index in enumerate(predicted_classes):
            class_name = f"Classe {class_index}"  # Ajuste conforme suas classes
            class_confidence = float(prediction[0][class_index])
            prediction_results[class_name] = class_confidence

        #Retorna os resultados originais em JSON;
        result_original = {'predictions': prediction_results}
            
        # Normaliza as probabilidades para que a soma seja igual a 1
        total_confidence = sum(prediction_results.values())
        normalized_results = {class_name: (class_confidence / total_confidence) for class_name, class_confidence in prediction_results.items()}

        # Retorna os resultados normalizados em JSON;
        result_normalizado = {'predictions': normalized_results}

        # Visualização Original;
        print('Resultados Originais:', result_original)
        # Visualização Normalizada;
        print('Resultados Normalizados:', result_normalizado)

        return jsonify(result_normalizado)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar dados base64: {str(e)}'}), 400
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug='true')
