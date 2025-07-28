// app.js - Sistema Completo de Recomendação de Receitas
class RecipeRecommender {
    constructor() {
        // Detecta automaticamente se está no desenvolvimento ou produção
        this.apiUrl = window.location.hostname === 'localhost' 
            ? 'http://localhost:5000/api'
            : '/api';
        this.initialized = false;
        this.googleDriveUrl = 'https://drive.google.com/drive/folders/1poHpksILFm9uIvBfJLogGzqbyUSQTFrt?usp=drive_link';
    }

    async init() {
        if (this.initialized) return true;

        try {
            // Primeiro verifica se o servidor tem os modelos
            let response = await fetch(`${this.apiUrl}/health`);
            let data = await response.json();
            
            if (data.status === 'ok' && data.models_loaded) {
                this.initialized = true;
                console.log("✅ Sistema inicializado com sucesso!");
                return true;
            }
            
            // Se não tem os modelos, solicita download do Google Drive
            console.log("📥 Baixando modelos do Google Drive...");
            response = await fetch(`${this.apiUrl}/download-models`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    drive_url: this.googleDriveUrl 
                })
            });

            if (!response.ok) {
                throw new Error(`Erro ao baixar modelos: ${response.statusText}`);
            }

            data = await response.json();
            
            if (data.success) {
                this.initialized = true;
                console.log("✅ Modelos baixados e sistema inicializado!");
                return true;
            } else {
                throw new Error(data.error || "Erro desconhecido ao baixar modelos");
            }
            
        } catch (error) {
            console.error("Erro na inicialização:", error);
            throw error;
        }
    }

    async suggest(ingredients) {
        if (!this.initialized) await this.init();

        try {
            const response = await fetch(`${this.apiUrl}/suggest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ingredients })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.recipes;
        } catch (error) {
            console.error("Erro na sugestão:", error);
            throw error;
        }
    }
}

// Instância global do recomendador
const recommender = new RecipeRecommender();

// Inicialização quando a página carrega
document.addEventListener('DOMContentLoaded', async () => {
    const inputEl = document.getElementById('ingredients-input');
    const buttonEl = document.getElementById('suggest-btn');
    const resultsEl = document.getElementById('results');

    // Verifica se o sistema está disponível
    try {
        resultsEl.innerHTML = '<p class="loading">Inicializando sistema...</p>';
        await recommender.init();
        buttonEl.disabled = false;
        resultsEl.innerHTML = '<p class="success">Sistema pronto! Digite seus ingredientes acima.</p>';
    } catch (error) {
        resultsEl.innerHTML = `<p class="error">Sistema indisponível: ${error.message}</p>`;
        buttonEl.disabled = true;
    }

    // Handler para o botão de sugerir
    buttonEl.addEventListener('click', async () => {
        const ingredients = inputEl.value.trim();
        if (!ingredients) {
            alert("Por favor, digite alguns ingredientes!");
            return;
        }

        try {
            buttonEl.disabled = true;
            buttonEl.textContent = "Buscando...";
            resultsEl.innerHTML = '<p class="loading">Analisando ingredientes...</p>';
            
            const recipes = await recommender.suggest(ingredients);
            displayResults(recipes);
        } catch (error) {
            resultsEl.innerHTML = `<p class="error">Erro: ${error.message}</p>`;
        } finally {
            buttonEl.disabled = false;
            buttonEl.textContent = "Sugerir Receitas";
        }
    });

    // Permite busca com Enter
    inputEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !buttonEl.disabled) {
            buttonEl.click();
        }
    });
});

// Exibe os resultados na página
function displayResults(recipes) {
    const resultsEl = document.getElementById('results');
    
    if (!recipes || recipes.length === 0) {
        resultsEl.innerHTML = '<p class="no-results">Nenhuma receita encontrada. Tente outros ingredientes!</p>';
        return;
    }

    resultsEl.innerHTML = recipes.map(recipe => `
        <div class="recipe-card">
            <h2 class="recipe-title">${recipe.title}</h2>
            <span class="match-percentage">${recipe.match}% compatível</span>
            
            <div class="ingredients-list">
                <div class="ingredients-section">
                    <h3>✅ Você tem:</h3>
                    <ul>
                        ${recipe.used_ingredients?.map(i => `<li class="ingredient has">${i}</li>`).join('') || '<li>Nenhum ingrediente principal</li>'}
                    </ul>
                </div>
                
                <div class="ingredients-section">
                    <h3>🛒 Precisa comprar:</h3>
                    <ul>
                        ${recipe.missing_ingredients?.map(i => `<li class="ingredient missing">${i}</li>`).join('') || '<li>Nada!</li>'}
                    </ul>
                </div>
            </div>
            
            ${recipe.directions ? `
                <div class="directions">
                    <h3>📝 Modo de preparo:</h3>
                    <p>${recipe.directions}</p>
                </div>
            ` : ''}
            
            ${recipe.link ? `<a href="${recipe.link}" target="_blank" class="recipe-link">🔗 Ver receita completa</a>` : ''}
        </div>
    `).join('');
}