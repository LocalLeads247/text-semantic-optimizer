document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('text-form');
    const input = document.getElementById('text-input');
    const optimizeBtn = document.getElementById('optimize-btn');
    const resultContainer = document.getElementById('result-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorAlert = document.getElementById('error-alert');
    
    let optimizationInProgress = false;
    
    const showLoading = () => {
        loadingSpinner.classList.remove('hidden');
        optimizeBtn.disabled = true;
        optimizationInProgress = true;
    };
    
    const hideLoading = () => {
        loadingSpinner.classList.add('hidden');
        optimizeBtn.disabled = false;
        optimizationInProgress = false;
    };
    
    const showError = (message) => {
        errorAlert.textContent = message;
        errorAlert.classList.remove('hidden');
        setTimeout(() => {
            errorAlert.classList.add('hidden');
        }, 5000);
    };
    
    const displayResults = (data) => {
        resultContainer.innerHTML = `
            <div class="space-y-4">
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-lg mb-2">Optimized Text:</h3>
                    <p class="text-gray-800">${data.optimized}</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h4 class="font-medium mb-2">Metrics:</h4>
                        <ul class="space-y-1">
                            ${Object.entries(data.metrics).map(([key, value]) => 
                                `<li><span class="text-gray-600">${key}:</span> ${value}</li>`
                            ).join('')}
                        </ul>
                    </div>
                    
                    <div class="bg-white p-4 rounded-lg shadow">
                        <h4 class="font-medium mb-2">Suggestions:</h4>
                        <ul class="list-disc list-inside space-y-1">
                            ${data.suggestions.map(suggestion => 
                                `<li class="text-gray-600">${suggestion}</li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
        resultContainer.classList.remove('hidden');
    };
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (optimizationInProgress) return;
        
        const text = input.value.trim();
        if (!text) {
            showError('Please enter some text to optimize');
            return;
        }
        
        showLoading();
        
        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: text,
                    optimization_level: document.getElementById('optimization-level').value,
                    preserve_keywords: document.getElementById('keywords').value
                        .split(',')
                        .map(k => k.trim())
                        .filter(k => k)
                })
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }
            
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            showError(`Failed to optimize text: ${error.message}`);
        } finally {
            hideLoading();
        }
    });
    
    // Add real-time character count
    input.addEventListener('input', () => {
        const charCount = document.getElementById('char-count');
        const count = input.value.length;
        charCount.textContent = `${count}/10000`;
        charCount.className = count > 10000 ? 'text-red-500' : 'text-gray-500';
    });
});
