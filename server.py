from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path
import gdown
import os
import zipfile

app = Flask(__name__)
CORS(app)

# Carrega modelos uma vez na inicialização
MODEL_DIR = 'model_optimized'
models = {}

def load_models():
    """Carrega todos os modelos na inicialização do servidor"""
    global models
    try:
        # Carrega pré-processadores
        preprocessors = joblib.load(f'{MODEL_DIR}/preprocessor.joblib')
        models['vectorizer'] = preprocessors['vectorizer']
        models['svd'] = preprocessors['svd']
        
        # Carrega modelo de similaridade
        models['similarity_model'] = joblib.load(f'{MODEL_DIR}/similarity_model.joblib')
        
        # Carrega dados normalizados
        data = np.load(f'{MODEL_DIR}/normalized_data.npz')
        models['normalized_data'] = data['data']
        
        # Carrega metadados
        models['metadata'] = pd.read_parquet(f'{MODEL_DIR}/metadata.parquet')
        
        print("✅ Modelos carregados com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return False

def preprocess_ingredient(ingredient):
    """Mesmo pré-processamento usado no treinamento"""
    ing = str(ingredient).lower()
    ing = re.sub(
        r'^\d+\s*[/\d\s]*\s*(c\.|tsp\.|tbsp\.|cup[s]*|ounce[s]*|g|kg|ml|l)\s*|[^\w\s]|\(.*?\)',
        '', 
        ing
    )
    return re.sub(r'\s+', ' ', ing).strip()

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('public', filename)

@app.route('/api/suggest', methods=['POST'])
def suggest_recipes():
    """Endpoint para sugestão de receitas"""
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', '')
        
        if not ingredients:
            return jsonify({'error': 'Ingredientes não fornecidos'}), 400
        
        # Pré-processa ingredientes
        processed = preprocess_ingredient(ingredients)
        
        # Vetorização
        X = models['vectorizer'].transform([processed])
        
        # Redução de dimensionalidade
        X_reduced = models['svd'].transform(X)
        
        # Normalização
        from sklearn.preprocessing import normalize
        X_normalized = normalize(X_reduced, norm='l2')
        
        # Busca por similaridade
        distances, indices = models['similarity_model'].kneighbors(X_normalized, n_neighbors=5)
        
        # Prepara resultados
        results = []
        user_ingredients = set(ingredients.lower().split(','))
        user_ingredients = {ing.strip() for ing in user_ingredients}
        
        for i, idx in enumerate(indices[0]):
            recipe = models['metadata'].iloc[idx]
            similarity = (1 - distances[0][i]) * 100
            
            # Analisa ingredientes
            recipe_ingredients = recipe['ingredients']
            used_ingredients = []
            missing_ingredients = []
            
            for ing in recipe_ingredients:
                ing_clean = preprocess_ingredient(ing)
                is_available = any(user_ing in ing_clean for user_ing in user_ingredients)
                
                if is_available:
                    used_ingredients.append(ing)
                else:
                    missing_ingredients.append(ing)
            
            results.append({
                'title': recipe['title'],
                'match': round(similarity),
                'used_ingredients': used_ingredients,
                'missing_ingredients': missing_ingredients,
                'directions': recipe['directions'],
                'link': recipe['link']
            })
        
        return jsonify({'recipes': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-models', methods=['POST'])
def download_models():
    try:
        data = request.get_json()
        drive_url = data.get('drive_url')
        
        if not drive_url:
            return jsonify({'success': False, 'error': 'URL do Google Drive não fornecida'}), 400
        
        # Diretório para salvar os modelos
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Extrair o ID da pasta do Google Drive
        folder_id = '1poHpksILFm9uIvBfJLogGzqbyUSQTFrt'
        
        # Download dos arquivos da pasta
        gdown.download_folder(f'https://drive.google.com/drive/folders/{folder_id}', 
                             output=models_dir, quiet=False, use_cookies=False)
        
        # Recarregar os modelos
        load_models()
        
        return jsonify({'success': True, 'message': 'Modelos baixados com sucesso'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Verifica se os modelos estão carregados"""
    return jsonify({
        'status': 'ok' if models else 'error',
        'models_loaded': bool(models)
    })

if __name__ == '__main__':
    if load_models():
        app.run(debug=True, port=5000)
    else:
        print("Falha ao carregar modelos. Certifique-se de que model.py foi executado primeiro.")