from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import re
from pathlib import Path
import gdown
import os
import sys

app = Flask(__name__)
CORS(app)

# Diretório base dos modelos
MODEL_DIR = '/tmp/model_optimized' if os.environ.get('VERCEL') else 'model_optimized'
models = {}

def install_dependencies():
    try:
        import joblib
        import pandas as pd
        from sklearn.preprocessing import normalize
        return True
    except ImportError:
        print("📦 Instalando dependências...")
        import subprocess
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "joblib==1.3.2", 
                "pandas==2.0.3", 
                "scikit-learn==1.3.0",
                "pyarrow==12.0.1"
            ], timeout=300)
            return True
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")
            return False

def load_models():
    global models
    if not install_dependencies():
        return False

    import joblib
    import pandas as pd

    try:
        if not os.path.exists(MODEL_DIR):
            print(f"❌ Diretório de modelos não encontrado: {MODEL_DIR}")
            return False

        print(f"📂 Carregando modelos do diretório: {MODEL_DIR}")
        preprocessors = joblib.load(f'{MODEL_DIR}/preprocessor.joblib')
        models['vectorizer'] = preprocessors['vectorizer']
        models['svd'] = preprocessors['svd']
        models['similarity_model'] = joblib.load(f'{MODEL_DIR}/similarity_model.joblib')
        data = np.load(f'{MODEL_DIR}/normalized_data.npz')
        models['normalized_data'] = data['data']
        models['metadata'] = pd.read_parquet(f'{MODEL_DIR}/metadata.parquet')
        print("✅ Modelos carregados com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        models.clear()
        return False

def preprocess_ingredient(ingredient):
    ing = str(ingredient).lower()
    ing = re.sub(
        r'^\d+\s*[/\d\s]*\s*(c\.|tsp\.|tbsp\.|cup[s]*|ounce[s]*|g|kg|ml|l)\s*|[^\w\s]|\(.*?\)',
        '', 
        ing
    )
    return re.sub(r'\s+', ' ', ing).strip()

@app.route('/')
def index():
    try:
        return send_from_directory('public', 'index.html')
    except:
        return jsonify({'message': 'SmartCook API is running'}), 200

@app.route('/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('public', filename)
    except:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/suggest', methods=['POST'])
def suggest_recipes():
    try:
        if not models:
            return jsonify({'error': 'Modelos não carregados. Tente novamente em alguns instantes.'}), 503

        from sklearn.preprocessing import normalize

        data = request.get_json()
        ingredients = data.get('ingredients', '')

        if not ingredients:
            return jsonify({'error': 'Ingredientes não fornecidos'}), 400

        processed = preprocess_ingredient(ingredients)
        X = models['vectorizer'].transform([processed])
        X_reduced = models['svd'].transform(X)
        X_normalized = normalize(X_reduced, norm='l2')

        distances, indices = models['similarity_model'].kneighbors(X_normalized, n_neighbors=5)

        results = []
        user_ingredients = {ing.strip() for ing in ingredients.lower().split(',')}

        for i, idx in enumerate(indices[0]):
            recipe = models['metadata'].iloc[idx]
            similarity = (1 - distances[0][i]) * 100
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

        folder_id = '1poHpksILFm9uIvBfJLogGzqbyUSQTFrt'
        temp_dir = '/tmp/model_optimized'
        os.makedirs(temp_dir, exist_ok=True)

        print("📥 Baixando modelos do Google Drive...")
        gdown.download_folder(
            f'https://drive.google.com/drive/folders/{folder_id}', 
            output=temp_dir, 
            quiet=False, 
            use_cookies=False
        )

        global MODEL_DIR
        MODEL_DIR = temp_dir
        success = load_models()

        if success:
            return jsonify({'success': True, 'message': 'Modelos baixados e carregados com sucesso'})
        else:
            return jsonify({'success': False, 'error': 'Erro ao carregar modelos após download'})

    except Exception as e:
        print(f"Erro no download: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    models_loaded = (
        bool(models) and 
        'vectorizer' in models and 
        'similarity_model' in models and 
        'metadata' in models
    )

    return jsonify({
        'status': 'ok' if models_loaded else 'needs_models',
        'models_loaded': models_loaded,
        'models_count': len(models) if models else 0
    })

# **FUNÇÃO HANDLER PARA O VERCEL - ESSENCIAL**
def handler(environ, start_response):
    """
    Função handler requerida pelo Vercel para aplicações Flask
    """
    return app(environ, start_response)

# Export da função para o Vercel
app = app

# Para desenvolvimento local
if __name__ == '__main__':
    # Tenta carregar os modelos na inicialização local
    possible_dirs = ['model_optimized', '/tmp/model_optimized', 'C:/tmp/model_optimized']
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            MODEL_DIR = dir_path
            print(f"📂 Encontrados modelos em: {MODEL_DIR}")
            load_models()
            break
    else:
        print("⚠️ Modelos não encontrados localmente. Serão baixados quando necessário.")
    
    app.run(debug=True)
