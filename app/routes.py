from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Input

app = Flask(__name__)

# Função para corrigir a camada de entrada do modelo
def load_model_with_custom_input(model_path):
    try:
        model = load_model(model_path)
        return model
    except TypeError as e:
        if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
            # Recria a camada de entrada
            input_layer = Input(shape=(150, 150, 3), dtype='float32', name='input_layer')
            # Carrega o restante do modelo
            model = load_model(model_path, custom_objects={'input_layer': input_layer})
            return model
        else:
            raise e

# Importar modelos treinados
model_fruto = load_model_with_custom_input('../model_fruto_w.h5')
model_class = load_model_with_custom_input('../model_2.h5')

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
        prediction_fruto = model_fruto.predict(processed_image) # Modelo inicial

        # Supondo que prediction_fruto seja um vetor de probabilidades
        prediction_values = prediction_fruto[0]  # Se prediction_fruto for um array 2D com uma única amostra

        # Cria o dicionário de previsões
        predictions = {
            "no_cacau": float(prediction_values[0]),
            "cacau": float(prediction_values[1])
        }

        print(predictions)

        # Condição aplicação dos modelos;
        if prediction_fruto[0][1] > 0.5: 
            # Modelo hierarquico: Se for cacau, usa o segundo modelo para classificação adicional
            prediction_class = model_class.predict(processed_image) # Modelo secundário

            # Obtém todas as classes previstas e suas confianças
            predicted_classes = np.argsort(prediction_class[0])[::-1]  # Obtém as classes ordenadas por confiança
            prediction_results = {}
            for i, class_index in enumerate(predicted_classes):
                class_name = f"Classe {class_index}"  # Ajuste conforme suas classes
                class_confidence = float(prediction_class[0][class_index])
                prediction_results[class_name] = class_confidence

            # Normaliza as probabilidades para que a soma seja igual a 1
            total_confidence = sum(prediction_results.values())
            normalized_results = {class_name: (class_confidence / total_confidence) for class_name, class_confidence in prediction_results.items()}

            # Visualização Original
            print('Resultados Originais:', prediction_results)
            # Visualização Normalizada
            print('Resultados Normalizados:', normalized_results)

            return jsonify({'predictions': normalized_results})
        
        # Inserir Mensagem de Fruto não é cacau, tente novamente, logo após o carregamento de 5s
        else:
            return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': f'Erro ao processar dados base64: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
