document.addEventListener('DOMContentLoaded', () => {
    // Configure API endpoint: prefer window.SENSECATCH_API_BASE if set
    const DEFAULT_API = '/analyze';
    const API_BASE = (typeof window !== 'undefined' && window.SENSECATCH_API_BASE) ? window.SENSECATCH_API_BASE : DEFAULT_API;
    const textInput = document.getElementById('text-input');
    const modelSelector = document.getElementById('model-selector');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultDisplay = document.getElementById('result-display');
    const historyList = document.getElementById('history-list');
    
    // Store analysis history
    const analysisHistory = [];
    
    // Banner helpers
    const infoBanner = document.getElementById('inline-info');
    const infoText = document.getElementById('info-text');
    const infoClose = document.getElementById('info-close');
    const FIRST_VISIT_KEY = 'sc_first_visit_shown_v1';
    const LAST_SUCCESS_TS = 'sc_last_success_ts_v1';
    const REWARM_SECS = 15 * 60; // 15 minutes

    function showInfo(message) {
        if (!infoBanner) return;
        infoText.textContent = message;
        infoBanner.classList.remove('hidden');
    }
    function hideInfo() {
        if (!infoBanner) return;
        infoBanner.classList.add('hidden');
    }
    if (infoClose) {
        infoClose.addEventListener('click', hideInfo);
    }

    // Initial banner for first click per session
    function maybeShowFirstVisitBanner() {
        if (!sessionStorage.getItem(FIRST_VISIT_KEY)) {
            showInfo('Waking up the machine learning models!\nFirst request may take up to ~40 seconds.');
            sessionStorage.setItem(FIRST_VISIT_KEY, '1');
        }
    }

    // Re-warm banner if app likely slept again (no success for 15+ mins)
    function maybeShowRewarmBanner() {
        const last = Number(sessionStorage.getItem(LAST_SUCCESS_TS) || '0');
        const now = Date.now() / 1000;
        if (!last || (now - last) > REWARM_SECS) {
            showInfo('The ML models went to sleep due to inactivity. Waking them up now…\n(this may take up to ~40 seconds)');
        }
    }

    // Handle analyze button click
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        const model = modelSelector.value;
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        // Show inline info if first visit and if rewarm needed
        maybeShowFirstVisitBanner();
        maybeShowRewarmBanner();

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        resultDisplay.innerHTML = '<p class="prompt">Analyzing your text...</p>';
        
        try {
            // Send request to server
            const response = await fetch(API_BASE, {
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
            // Success - record timestamp and hide banner
            sessionStorage.setItem(LAST_SUCCESS_TS, String(Math.floor(Date.now()/1000)));
            hideInfo();
            
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
        
        // Create HTML for important words
        let wordChips = '';
        if (result.important_words && result.important_words.length > 0) {
            wordChips = result.important_words.map(wordInfo => {
                const chipClass = wordInfo.sentiment === 'positive' ? 'positive' : 'negative';
                // Add strikethrough styling for negated words
                const negatedStyle = wordInfo.negated ? 'text-decoration: line-through;' : '';
                return `<span class="word-chip ${chipClass}" style="${negatedStyle}" title="${wordInfo.negated ? 'Negated' : ''}">${wordInfo.word}</span>`;
            }).join('');
        } else {
            wordChips = '<span class="no-words">No influential words found</span>';
        }
        
        resultDisplay.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <span class="sentiment-label ${sentimentClass}">${result.sentiment}</span>
                    <span class="confidence">${result.confidence}% confidence</span>
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-bar ${sentimentClass}" style="width: ${result.confidence}%"></div>
                </div>
                
                <div class="result-text">"${result.text}"</div>
                
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
            case 'linear_svc':
                return 'Linear SVC';
            case 'nbsvm':
                return 'NBSVM';
            case 'distilbert':
                return 'DistilBERT (fine-tuned)';
            case 'stack':
                return 'Stacked Ensemble';
            case 'rule_based':
                return 'Rule-based (linguistic)';
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
    
    // Set up clear history button
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    clearHistoryBtn.addEventListener('click', () => {
        if (analysisHistory.length === 0) {
            return; // Nothing to clear
        }
        
        if (confirm('Are you sure you want to clear your analysis history?')) {
            // Clear history array
            analysisHistory.length = 0;
            
            // Update display
            updateHistoryDisplay();
        }
    });

    // Optional: background warm-up ping to reduce free-tier cold start delay
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        fetch(API_BASE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: 'warmup', model: 'naive_bayes' }),
            signal: controller.signal
        }).finally(() => clearTimeout(timeoutId));
    } catch (_) {
        // ignore warm-up failures
    }
});