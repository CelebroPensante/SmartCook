from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import ast
import re
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Configura CORS para permitir requests do GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de request
class IngredientsRequest(BaseModel):
    ingredients: str

# Carrega os modelos
try:
    vectorizer = joblib.load('model/vectorizer.joblib')
    nn = joblib.load('model/nn_model.joblib')
    metadata = joblib.load('model/metadata.joblib')
except Exception as e:
    raise RuntimeError(f"Erro ao carregar modelos: {str(e)}")

def preprocess_ingredient(ingredient: str) -> str:
    """Função idêntica à usada no treinamento"""
    ing = str(ingredient).lower()
    ing = re.sub(r'^\d+\s*[/\d\s]*\s*(c\.|tsp\.|tbsp\.|cup[s]*|ounce[s]*|g|kg|ml|l)\s*', '', ing)
    ing = re.sub(r'\(.*?\)|[^\w\s]', '', ing)
    ing = re.sub(r'\s+', ' ', ing).strip()
    return ing

@app.post("/suggest")
async def suggest_recipes(request: IngredientsRequest):
    try:
        user_ingredients = [preprocess_ingredient(ing.strip()) for ing in request.ingredients.split(',')]
        user_ingredients_set = set(user_ingredients)
        
        input_vec = vectorizer.transform([' '.join(user_ingredients)])
        distances, indices = nn.kneighbors(input_vec, n_neighbors=5)
        
        suggestions = []
        for idx in indices[0]:
            recipe = metadata[idx]
            recipe_ingredients = ast.literal_eval(recipe['ingredients'])
            
            exact_matches = []
            missing = []
            
            for ing in recipe_ingredients:
                ing_processed = preprocess_ingredient(ing)
                if ing_processed in user_ingredients_set:
                    exact_matches.append(ing)
                else:
                    missing.append(ing)
            
            match_percentage = round(len(exact_matches) / len(recipe_ingredients) * 100)
            
            suggestions.append({
                "title": recipe['title'],
                "match": match_percentage,
                "used": exact_matches,
                "missing": missing,
                "directions": recipe.get('directions', ''),
                "link": recipe.get('link', '')
            })
        
        return {"recipes": suggestions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))