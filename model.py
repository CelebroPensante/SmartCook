import pandas as pd
import ast
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib
from pathlib import Path
import sys

# Configurações
DATASET_PATH = 'recipes_data.csv'
MODEL_DIR = 'model_optimized'
CHUNKSIZE = 50000
N_RECIPES = 5
N_COMPONENTS = 100  # Para redução de dimensionalidade

Path(MODEL_DIR).mkdir(exist_ok=True)

def preprocess_ingredient(ingredient):
    """Versão otimizada do pré-processamento"""
    ing = str(ingredient).lower()
    # Remove medidas, unidades e caracteres especiais em uma única passada
    ing = re.sub(
        r'^\d+\s*[/\d\s]*\s*(c\.|tsp\.|tbsp\.|cup[s]*|ounce[s]*|g|kg|ml|l)\s*|[^\w\s]|\(.*?\)',
        '', 
        ing
    )
    return re.sub(r'\s+', ' ', ing).strip()

def train_optimized_model():
    """Treina modelos com otimizações de espaço"""
    try:
        # 1. Vectorizer mais eficiente
        vectorizer = HashingVectorizer(
            n_features=2**18, 
            alternate_sign=False,
            dtype=np.float32  # Reduz tamanho pela metade
        )
        
        # 2. Estrutura para armazenar dados esparsos
        X_list = []
        metadata = []
        processed_count = 0
        
        print("Iniciando processamento em chunks...")
        for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNKSIZE):
            print(f"Processando chunk com {len(chunk)} receitas...")
            
            if 'ingredients' not in chunk.columns:
                print("AVISO: Coluna 'ingredients' não encontrada no chunk. Pulando...")
                continue
                
            try:
                # Pré-processamento em lote (mais rápido)
                chunk['processed'] = chunk['ingredients'].apply(preprocess_ingredient)
                X_chunk = vectorizer.transform(chunk['processed'])
                
                # Converte para CSR imediatamente para economizar memória
                X_list.append(X_chunk.tocsr())
                processed_count += len(chunk)
                
                # Armazena apenas metadados essenciais
                for _, row in chunk.iterrows():
                    try:
                        ingredients = ast.literal_eval(row['ingredients'])
                        metadata.append({
                            'title': str(row.get('title', ''))[:100],  # Limita tamanho
                            'ingredients': ingredients,
                            'directions': str(row.get('directions', ''))[:300],  # Limita
                            'link': str(row.get('link', ''))[:200]
                        })
                    except Exception as e:
                        print(f"Erro ao processar metadados: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Erro grave no processamento do chunk: {str(e)}")
                continue
        
        if not X_list:
            raise ValueError("Nenhum dado válido foi processado")
        
        print(f"\n✅ Processamento concluído. Total de receitas: {processed_count}")
        
        # 3. Concatena todas as matrizes esparsas
        print("Concatenando matrizes esparsas...")
        X = vstack(X_list) if len(X_list) > 1 else X_list[0]
        
        # 4. Redução de dimensionalidade
        print("Aplicando redução de dimensionalidade...")
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
        X_reduced = svd.fit_transform(X)
        
        # 5. Normaliza os dados para usar cosine similarity
        print("Normalizando dados...")
        X_normalized = normalize(X_reduced, norm='l2')
        
        # 6. Treina modelo de similaridade com NearestNeighbors
        print("Treinando modelo de similaridade...")
        model = NearestNeighbors(
            n_neighbors=N_RECIPES + 1,  # +1 porque inclui a própria receita
            metric='cosine',
            algorithm='brute',  # Mais eficiente para cosine
            n_jobs=-1
        )
        model.fit(X_normalized)
        
        # 7. Salva modelos comprimidos
        print("\nSalvando modelos otimizados...")
        
        # Pré-processadores
        joblib.dump(
            {'vectorizer': vectorizer, 'svd': svd},
            f'{MODEL_DIR}/preprocessor.joblib',
            compress=('zlib', 3)  # Compactação máxima
        )
        print(f"✔ Pré-processadores salvos em {MODEL_DIR}/preprocessor.joblib")
        
        # Modelo de similaridade
        joblib.dump(
            model,
            f'{MODEL_DIR}/similarity_model.joblib',
            compress=('zlib', 3)
        )
        print(f"✔ Modelo de similaridade salvo em {MODEL_DIR}/similarity_model.joblib")
        
        # Dados normalizados para uso posterior
        np.savez_compressed(
            f'{MODEL_DIR}/normalized_data.npz',
            data=X_normalized
        )
        print(f"✔ Dados normalizados salvos em {MODEL_DIR}/normalized_data.npz")
        
        # Metadados em formato binário eficiente
        pd.DataFrame(metadata).to_parquet(
            f'{MODEL_DIR}/metadata.parquet',
            engine='pyarrow',
            compression='brotli'
        )
        print(f"✔ Metadados salvos em {MODEL_DIR}/metadata.parquet")
        
        print("\n🎉 Modelos otimizados salvos com sucesso!")
        print(f"Total de receitas processadas: {processed_count}")
        print(f"Dimensões finais: {X_normalized.shape}")
        print(f"Tamanho do modelo: {N_RECIPES} receitas similares")
        
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    train_optimized_model()