// script.js atualizado
const API_URL = "https://sua-api.onrender.com"; // Substitua pela sua URL

async function suggestRecipes() {
    const input = document.getElementById('ingredients-input').value;
    if (!input.trim()) {
        alert("Por favor, digite alguns ingredientes!");
        return;
    }

    try {
        // Mostra loading
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<p class="loading">Buscando receitas...</p>';

        const response = await fetch(`${API_URL}/suggest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ingredients: input })
        });

        if (!response.ok) {
            throw new Error('Erro ao buscar receitas');
        }

        const data = await response.json();
        displayResults(data.recipes);
    } catch (error) {
        console.error("Erro:", error);
        document.getElementById('results').innerHTML = 
            `<p class="error">Erro ao buscar receitas: ${error.message}</p>`;
    }
}
// Exibe os resultados na página
function displayResults(recipes) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (recipes.length === 0) {
        resultsDiv.innerHTML = '<p class="no-results">Nenhuma receita encontrada. Tente outros ingredientes!</p>';
        return;
    }

    recipes.forEach(recipe => {
        const recipeCard = document.createElement('div');
        recipeCard.className = 'recipe-card';
        
        recipeCard.innerHTML = `
            <h2 class="recipe-title">${recipe.title} <span class="match-percentage">${recipe.match}%</span></h2>
            
            <div class="ingredients-grid">
                <div class="ingredients-used">
                    <h3>✅ Você tem:</h3>
                    <ul>
                        ${recipe.used.map(ing => `<li class="ingredient has">✔ ${ing}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="ingredients-missing">
                    <h3>❌ Faltando:</h3>
                    <ul>
                        ${recipe.missing.map(ing => `<li class="ingredient missing">✖ ${ing}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            ${recipe.directions ? `<div class="directions"><p>📝 <strong>Modo de preparo:</strong> ${recipe.directions}</p></div>` : ''}
            
            ${recipe.link ? `<a href="${recipe.link}" target="_blank" class="recipe-link">🔗 Ver receita completa</a>` : ''}
        `;
        
        resultsDiv.appendChild(recipeCard);
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    document.getElementById('suggest-btn').addEventListener('click', suggestRecipes);
});