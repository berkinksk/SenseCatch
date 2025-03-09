document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const modelSelector = document.getElementById('model-selector');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultDisplay = document.getElementById('result-display');
    const historyList = document.getElementById('history-list');
    
    // Store analysis history
    const analysisHistory = [];
    
    // Handle analyze button click
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        const model = modelSelector.value;
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        resultDisplay.innerHTML = '<p class="prompt">Analyzing your text...</p>';
        
        try {
            // Send request to server
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, model })
            });
            
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            
            const result = await response.json();
            
            // Display result
            displayResult(result);
            
            // Add to history
            addToHistory(result);
        } catch (error) {
            console.error('Error analyzing text:', error);
            resultDisplay.innerHTML = `
                <div class="result-card">
                    <p>Error analyzing text: ${error.message || 'Unknown error'}. Please try again.</p>
                </div>
            `;
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Sentiment';
        }
    });
    
    // Display analysis result
    function displayResult(result) {
        const sentimentClass = result.sentiment === 'Positive' ? 'positive' : 'negative';
        
        // Highlight important words in the text
        let highlightedText = result.text;
        
        // Create HTML for important words
        const wordChips = result.important_words.map(wordInfo => {
            const chipClass = wordInfo.sentiment === 'positive' ? 'positive' : 'negative';
            return `<span class="word-chip ${chipClass}">${wordInfo.word}</span>`;
        }).join('');
        
        resultDisplay.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="sentiment-label ${sentimentClass}">${result.sentiment}</span>
                    <span class="confidence">${result.confidence}% confidence</span>
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-bar ${sentimentClass}" style="width: ${result.confidence}%"></div>
                </div>
                
                <div class="result-text">"${highlightedText}"</div>
                
                <div class="important-words">
                    <h4>Influential Words:</h4>
                    <div class="word-chips">
                        ${wordChips}
                    </div>
                </div>
                
                <p class="model-type">Model: ${formatModelName(result.model)}</p>
            </div>
        `;
    }
    
    // Add result to history
    function addToHistory(result) {
        // Add to history array (limit to 10 items)
        analysisHistory.unshift(result);
        if (analysisHistory.length > 10) {
            analysisHistory.pop();
        }
        
        // Update history display
        updateHistoryDisplay();
    }
    
    // Update history display
    function updateHistoryDisplay() {
        // Clear "no history" message
        historyList.innerHTML = '';
        
        if (analysisHistory.length === 0) {
            historyList.innerHTML = '<p class="no-history">Previous analyses will appear here.</p>';
            return;
        }
        
        analysisHistory.forEach(item => {
            const sentimentClass = item.sentiment === 'Positive' ? 'positive' : 'negative';
            
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <div class="history-sentiment ${sentimentClass}">
                    ${item.sentiment} (${item.confidence}%)
                </div>
                <div class="history-text">${truncateText(item.text, 60)}</div>
                <div class="history-model">Model: ${formatModelName(item.model)}</div>
            `;
            
            // Add click event to load this analysis again
            historyItem.addEventListener('click', () => {
                textInput.value = item.text;
                modelSelector.value = item.model;
                
                // Scroll to input
                textInput.scrollIntoView({ behavior: 'smooth' });
                textInput.focus();
            });
            
            historyList.appendChild(historyItem);
        });
    }
    
    // Helper function to format model name
    function formatModelName(modelKey) {
        switch(modelKey) {
            case 'naive_bayes':
                return 'Naive Bayes';
            case 'logistic_regression':
                return 'Logistic Regression';
            default:
                return modelKey;
        }
    }
    
    // Helper function to truncate text
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }
    
    // Add focus to the text input on page load
    textInput.focus();
});