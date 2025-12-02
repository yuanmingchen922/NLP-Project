/**
 * Advanced NLP Features JavaScript
 * Handles entity extraction, keyword extraction, summarization, and aspect analysis
 */

// Extract Named Entities
document.getElementById('extractEntitiesBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/entities', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3"><div class="card-header"><h5 class="mb-0">Named Entity Recognition</h5></div><div class="card-body">';

        if (data.entities && Object.keys(data.entities).length > 0) {
            html += '<p><strong>Found ' + data.total_entities + ' entities:</strong></p>';
            html += '<div class="row">';

            for (const [type, entities] of Object.entries(data.entities)) {
                html += '<div class="col-md-6 mb-3">';
                html += '<h6 class="text-yelp-red">' + type + ' (' + entities.length + ')</h6>';
                html += '<ul class="list-group list-group-flush">';
                entities.slice(0, 10).forEach(entity => {
                    html += '<li class="list-group-item py-1">' + entity + '</li>';
                });
                if (entities.length > 10) {
                    html += '<li class="list-group-item py-1 text-muted">+ ' + (entities.length - 10) + ' more</li>';
                }
                html += '</ul></div>';
            }
            html += '</div>';
        } else {
            html += '<p class="text-muted">No entities found.</p>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Entity extraction error');
    });
});

// Extract Keywords
document.getElementById('extractKeywordsBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/keywords', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text, top_n: 10 })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3"><div class="card-header"><h5 class="mb-0">Keyword Extraction</h5></div><div class="card-body">';

        if (data.keywords && data.keywords.length > 0) {
            html += '<h6>Top Keywords:</h6>';
            html += '<div class="d-flex flex-wrap gap-2 mb-3">';
            data.keywords.forEach(kw => {
                const size = Math.max(1, Math.min(3, Math.ceil(kw.score / 5)));
                html += '<span class="badge bg-primary" style="font-size: ' + size + 'em; padding: 0.5em;">' + kw.word + ' (' + kw.score + ')</span>';
            });
            html += '</div>';
        }

        if (data.aspects && Object.keys(data.aspects).length > 0) {
            html += '<h6 class="mt-3">Aspects Mentioned:</h6>';
            for (const [aspect, mentions] of Object.entries(data.aspects)) {
                html += '<div class="mb-2"><strong class="text-yelp-red">' + aspect.charAt(0).toUpperCase() + aspect.slice(1) + ':</strong> ';
                html += '<span class="text-muted">' + mentions.length + ' mentions</span></div>';
            }
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Keyword extraction error');
    });
});

// Summarize Text
document.getElementById('summarizeBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/summarize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text, num_sentences: 2 })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3"><div class="card-header"><h5 class="mb-0">Text Summarization</h5></div><div class="card-body">';

        html += '<div class="alert alert-info"><strong>Summary:</strong><br>' + data.summary + '</div>';

        html += '<p><small class="text-muted">Reduced from ' + data.original_length + ' words to ' + data.summary_length + ' words</small></p>';

        if (data.key_points && data.key_points.length > 0) {
            html += '<h6 class="mt-3">Key Points:</h6><ul class="list-group">';
            data.key_points.forEach(point => {
                html += '<li class="list-group-item">' + point + '</li>';
            });
            html += '</ul>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Summarization error');
    });
});

// Aspect-Based Sentiment Analysis
document.getElementById('aspectAnalysisBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/aspects', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3"><div class="card-header"><h5 class="mb-0">Aspect-Based Sentiment Analysis</h5></div><div class="card-body">';

        if (data.aspect_sentiments) {
            html += '<div class="row">';
            for (const [aspect, analysis] of Object.entries(data.aspect_sentiments)) {
                html += '<div class="col-md-6 mb-3">';
                html += '<div class="card">';
                html += '<div class="card-body">';
                html += '<h6 class="text-yelp-red text-capitalize">' + aspect + '</h6>';

                if (analysis.mentions > 0) {
                    const sentimentClass = analysis.overall_sentiment === 'positive' ? 'success' :
                                          analysis.overall_sentiment === 'negative' ? 'danger' : 'warning';
                    html += '<span class="badge bg-' + sentimentClass + '">' + analysis.overall_sentiment.toUpperCase() + '</span> ';
                    html += '<span class="text-muted">(' + (analysis.confidence * 100).toFixed(0) + '% confidence)</span>';
                    html += '<p class="mt-2 mb-0"><small>' + analysis.mentions + ' mention(s)</small></p>';
                } else {
                    html += '<span class="text-muted">Not mentioned</span>';
                }

                html += '</div></div></div>';
            }
            html += '</div>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Aspect analysis error');
    });
});

// Comprehensive Analysis
document.getElementById('comprehensiveBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/comprehensive', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="alert alert-success"><h5>Comprehensive Analysis Complete</h5></div>';

        // Summary
        if (data.summary) {
            html += '<div class="card mb-3"><div class="card-header bg-yelp-red text-white"><h6 class="mb-0">Summary</h6></div>';
            html += '<div class="card-body">' + data.summary + '</div></div>';
        }

        // Keywords
        if (data.keywords && data.keywords.length > 0) {
            html += '<div class="card mb-3"><div class="card-header"><h6 class="mb-0">Top Keywords</h6></div><div class="card-body">';
            html += '<div class="d-flex flex-wrap gap-2">';
            data.keywords.slice(0, 10).forEach(([word, score]) => {
                html += '<span class="badge bg-primary">' + word + '</span>';
            });
            html += '</div></div></div>';
        }

        // Entities
        if (data.entities && Object.keys(data.entities).length > 0) {
            html += '<div class="card mb-3"><div class="card-header"><h6 class="mb-0">Named Entities</h6></div><div class="card-body">';
            for (const [type, entities] of Object.entries(data.entities)) {
                html += '<strong>' + type + ':</strong> ' + entities.slice(0, 5).join(', ');
                if (entities.length > 5) html += '...';
                html += '<br>';
            }
            html += '</div></div>';
        }

        // Aspects
        if (data.aspect_sentiments) {
            html += '<div class="card mb-3"><div class="card-header"><h6 class="mb-0">Aspect Sentiments</h6></div><div class="card-body"><div class="row">';
            for (const [aspect, analysis] of Object.entries(data.aspect_sentiments)) {
                if (analysis.mentions > 0) {
                    const sentimentClass = analysis.overall_sentiment === 'positive' ? 'success' :
                                          analysis.overall_sentiment === 'negative' ? 'danger' : 'warning';
                    html += '<div class="col-6 mb-2"><span class="text-capitalize">' + aspect + ':</span> ';
                    html += '<span class="badge bg-' + sentimentClass + ' ms-2">' + analysis.overall_sentiment + '</span></div>';
                }
            }
            html += '</div></div></div>';
        }

        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Comprehensive analysis error');
    });
});

// Helper functions
function createAdvancedResultsDiv() {
    const section = document.getElementById('resultsSection');
    let div = document.getElementById('advancedResults');
    if (!div) {
        div = document.createElement('div');
        div.id = 'advancedResults';
        div.className = 'mt-3';
        section.appendChild(div);
    }
    return div;
}

function showLoading() {
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingIndicator').style.display = 'none';
}

function showResults() {
    document.getElementById('resultsSection').style.display = 'block';
}
