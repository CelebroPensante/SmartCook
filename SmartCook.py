import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack
import joblib
from pathlib import Path

# Configurações
DATASET_PATH = 'recipes_data.csv'
MODEL_DIR = 'model'
CHUNKSIZE = 50000
N_RECIPES = 5

Path(MODEL_DIR).mkdir(exist_ok=True)

def preprocess_ingredient(ingredient):
    """Pré-processamento rigoroso mantendo a essência do ingrediente"""
    # Converte para string e lowercase
    ing = str(ingredient).lower()
    # Remove medidas e unidades
    ing = re.sub(r'^\d+\s*[/\d\s]*\s*(c\.|tsp\.|tbsp\.|cup[s]*|ounce[s]*|g|kg|ml|l)\s*', '', ing)
    # Remove texto entre parênteses e caracteres especiais
    ing = re.sub(r'\(.*?\)|[^\w\s]', '', ing)
    # Remove espaços extras e plural
    ing = re.sub(r'\s+', ' ', ing).strip()
    return ing

def train_model():
    """Treina o modelo e salva os arquivos necessários"""
    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False)
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
    
    X_list = []
    metadata = []
    
    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNKSIZE):
        print(f"Processando {len(chunk)} receitas...")
        
        # Garante que a coluna de ingredientes existe
        if 'ingredients' not in chunk.columns:
            continue
            
        chunk['processed'] = chunk['ingredients'].apply(preprocess_ingredient)
        X_chunk = vectorizer.transform(chunk['processed'])
        X_list.append(X_chunk)
        
        for _, row in chunk.iterrows():
            try:
                ingredients = ast.literal_eval(row['ingredients'])
            except:
                continue
                
            metadata.append({
                'title': row.get('title', ''),
                'ingredients': ingredients,
                'directions': row.get('directions', ''),
                'link': row.get('link', ''),
                'processed': preprocess_ingredient(row['ingredients'])
            })
    
    if not X_list:
        raise ValueError("Nenhum dado válido foi processado")
    
    X = vstack(X_list)
    nn.fit(X)
    
    joblib.dump(vectorizer, f'{MODEL_DIR}/vectorizer.joblib')
    joblib.dump(nn, f'{MODEL_DIR}/nn_model.joblib')
    joblib.dump(metadata, f'{MODEL_DIR}/metadata.joblib')
    print("Modelo treinado com sucesso!")

def load_model():
    """Carrega o modelo treinado"""
    return (
        joblib.load(f'{MODEL_DIR}/vectorizer.joblib'),
        joblib.load(f'{MODEL_DIR}/nn_model.joblib'),
        joblib.load(f'{MODEL_DIR}/metadata.joblib')
    )

def suggest_recipes(user_input, vectorizer, nn, metadata):
    """Sugere receitas com correspondência exata de ingredientes"""
    user_ingredients = [preprocess_ingredient(ing.strip()) for ing in user_input.split(',')]
    user_ingredients_set = set(user_ingredients)
    
    input_vec = vectorizer.transform([' '.join(user_ingredients)])
    distances, indices = nn.kneighbors(input_vec, n_neighbors=100)
    
    suggestions = []
    for idx in indices[0]:
        if idx >= len(metadata):
            continue
            
        recipe = metadata[idx]
        
        # Verifica se a receita tem a estrutura esperada
        if not all(key in recipe for key in ['title', 'ingredients']):
            continue
            
        try:
            recipe_ingredients = recipe['ingredients']
            if not isinstance(recipe_ingredients, list):
                recipe_ingredients = ast.literal_eval(recipe_ingredients)
        except:
            continue
            
        # Ignora receitas muito simples
        if len(recipe_ingredients) < 3:
            continue
            
        # Verificação exata
        exact_matches = []
        missing = []
        
        for ing in recipe_ingredients:
            ing_processed = preprocess_ingredient(ing)
            if ing_processed in user_ingredients_set:
                exact_matches.append(ing)
            else:
                missing.append(ing)
        
        # Calcula porcentagem de match
        match_percentage = round(len(exact_matches) / len(recipe_ingredients) * 100)
        
        # Filtro mínimo
        if len(exact_matches) < max(2, len(user_ingredients) * 0.3):
            continue
            
        suggestions.append({
            'title': recipe['title'],
            'match': match_percentage,
            'used': exact_matches,
            'missing': missing,
            'total_ingredients': len(recipe_ingredients),
            'directions': recipe.get('directions', ''),
            'link': recipe.get('link', '')
        })
    
    # Ordenação final
    suggestions.sort(key=lambda x: (-x['match'], -len(x['used'])))
    
    return suggestions[:N_RECIPES]

def main():
    """Interface principal"""
    if not all(Path(f'{MODEL_DIR}/{f}').exists() for f in ['vectorizer.joblib', 'nn_model.joblib', 'metadata.joblib']):
        print("Modelo não encontrado. Iniciando treinamento...")
        train_model()
    
    vectorizer, nn, metadata = load_model()
    
    print("\n🔍 Sugestor de Receitas - Digite seus ingredientes (separados por vírgula)")
    print("Exemplo: ovos, farinha, açúcar, leite\n")
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break
            
        if not user_input:
            print("Por favor, digite alguns ingredientes.")
            continue
            
        suggestions = suggest_recipes(user_input, vectorizer, nn, metadata)
        
        if not suggestions:
            print("Nenhuma receita adequada encontrada. Tente outros ingredientes.")
            continue
            
        print(f"\n🍳 Receitas sugeridas para: {user_input}")
        for i, recipe in enumerate(suggestions, 1):
            print(f"\n{i}. {recipe['title']} ({recipe['match']}% compatível)")
            print(f"   📌 Ingredientes usados ({len(recipe['used'])}/{recipe['total_ingredients']}):")
            for ing in recipe['used']:
                print(f"    ✔ {ing}")
            
            if recipe['missing']:
                print("   ❌ Ingredientes faltando:")
                for ing in recipe['missing']:
                    print(f"    ✖ {ing}")
            
            if recipe['directions']:
                print(f"\n   📝 Preparo: {recipe['directions'][:200]}...")
            if recipe['link']:
                print(f"   🔗 Mais info: {recipe['link']}")

if __name__ == "__main__":
    main()