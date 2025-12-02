/**
 * Professional Analysis Features JavaScript
 * Handles quality scoring, spam detection, and credibility analysis
 */

// Quality Score Analysis
document.getElementById('qualityScoreBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/quality', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3">';
        html += '<div class="card-header bg-primary text-white"><h5 class="mb-0">Review Quality Analysis</h5></div>';
        html += '<div class="card-body">';

        // Overall Score
        html += '<div class="row mb-4">';
        html += '<div class="col-md-12 text-center">';
        html += '<h2 class="display-4">' + data.overall_score + '</h2>';
        html += '<p class="lead">Quality Score</p>';

        const badgeClass = data.quality_class === 'high' ? 'success' :
                          data.quality_class === 'medium-high' ? 'info' :
                          data.quality_class === 'medium' ? 'warning' : 'danger';
        html += '<span class="badge bg-' + badgeClass + ' fs-5">' + data.quality_label + '</span>';
        html += '</div></div>';

        // Metrics
        html += '<div class="row mb-3">';
        html += '<div class="col-md-12"><h6 class="border-bottom pb-2">Detailed Metrics</h6></div>';
        for (const [metric, score] of Object.entries(data.metrics)) {
            const metricName = metric.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            html += '<div class="col-md-6 mb-2">';
            html += '<div class="d-flex justify-content-between align-items-center">';
            html += '<span>' + metricName + '</span>';
            html += '<div class="progress" style="width: 60%; height: 20px;">';
            html += '<div class="progress-bar" role="progressbar" style="width: ' + score + '%">';
            html += score.toFixed(0) + '</div></div></div></div>';
        }
        html += '</div>';

        // Insights
        if (data.insights && data.insights.length > 0) {
            html += '<div class="mt-3">';
            html += '<h6 class="border-bottom pb-2">Insights & Recommendations</h6>';
            html += '<ul class="list-group list-group-flush">';
            data.insights.forEach(insight => {
                html += '<li class="list-group-item"><i class="bi bi-lightbulb"></i> ' + insight + '</li>';
            });
            html += '</ul></div>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Quality analysis error');
    });
});

// Spam Detection
document.getElementById('spamDetectionBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/spam', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3">';
        html += '<div class="card-header"><h5 class="mb-0">Spam Detection Analysis</h5></div>';
        html += '<div class="card-body">';

        // Risk Level
        const riskClass = data.risk_level === 'none' ? 'success' :
                         data.risk_level === 'low' ? 'info' :
                         data.risk_level === 'medium' ? 'warning' : 'danger';

        html += '<div class="row mb-4">';
        html += '<div class="col-md-6 text-center">';
        html += '<h2>' + data.spam_score + '%</h2>';
        html += '<p class="text-muted">Spam Probability</p>';
        html += '</div>';
        html += '<div class="col-md-6 text-center">';
        html += '<h3 class="badge bg-' + riskClass + ' p-3">' + data.classification + '</h3>';
        html += '<p class="text-muted mt-2">Risk Level: ' + data.risk_level.toUpperCase() + '</p>';
        html += '</div></div>';

        // Flags
        if (data.flags && data.flags.length > 0) {
            html += '<div class="alert alert-warning">';
            html += '<strong>Warning Signs Detected:</strong>';
            html += '<ul class="mb-0 mt-2">';
            data.flags.forEach(flag => {
                html += '<li>' + flag + '</li>';
            });
            html += '</ul></div>';
        } else {
            html += '<div class="alert alert-success">No spam indicators detected</div>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Spam detection error');
    });
});

// Credibility Analysis
document.getElementById('credibilityBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/credibility', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="card mb-3">';
        html += '<div class="card-header"><h5 class="mb-0">Credibility Analysis</h5></div>';
        html += '<div class="card-body">';

        // Credibility Score
        const credClass = data.credibility_level === 'high' ? 'success' :
                         data.credibility_level === 'medium-high' ? 'info' :
                         data.credibility_level === 'medium' ? 'warning' : 'danger';

        html += '<div class="row mb-4">';
        html += '<div class="col-md-6 text-center">';
        html += '<h2>' + data.credibility_score + '%</h2>';
        html += '<p class="text-muted">Credibility Score</p>';
        html += '</div>';
        html += '<div class="col-md-6 text-center">';
        html += '<h4 class="badge bg-' + credClass + ' p-3">' + data.classification + '</h4>';
        html += '</div></div>';

        // Flags
        if (data.flags && data.flags.length > 0) {
            html += '<div class="alert alert-info">';
            html += '<strong>Analysis Notes:</strong>';
            html += '<ul class="mb-0 mt-2">';
            data.flags.forEach(flag => {
                html += '<li>' + flag + '</li>';
            });
            html += '</ul></div>';
        }

        html += '</div></div>';
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Credibility analysis error');
    });
});

// Professional Report (Comprehensive)
document.getElementById('professionalBtn')?.addEventListener('click', function() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) {
        alert('Please enter review text');
        return;
    }

    showLoading();

    fetch('/api/analyze/professional', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResults();

        const resultDiv = document.getElementById('advancedResults') || createAdvancedResultsDiv();

        let html = '<div class="alert alert-primary"><h5>Professional Analysis Report</h5></div>';

        // Overall Trustworthiness
        html += '<div class="card mb-3">';
        html += '<div class="card-body text-center">';
        html += '<h2 class="display-4">' + data.overall_trustworthiness + '%</h2>';
        html += '<p class="lead">Overall Trustworthiness Score</p>';
        html += '<p class="text-muted">' + data.recommendation + '</p>';
        html += '</div></div>';

        // Quality Analysis
        html += '<div class="card mb-3">';
        html += '<div class="card-header"><h6 class="mb-0">Quality Assessment</h6></div>';
        html += '<div class="card-body">';
        html += '<div class="row">';
        html += '<div class="col-md-4 text-center">';
        html += '<h4>' + data.quality.overall_score + '</h4>';
        html += '<p class="text-muted">Quality Score</p>';
        html += '<span class="badge bg-secondary">' + data.quality.quality_label + '</span>';
        html += '</div>';
        html += '<div class="col-md-8">';
        if (data.quality.insights && data.quality.insights.length > 0) {
            html += '<ul class="list-unstyled">';
            data.quality.insights.slice(0, 3).forEach(insight => {
                html += '<li class="mb-1"><small>' + insight + '</small></li>';
            });
            html += '</ul>';
        }
        html += '</div></div></div></div>';

        // Spam Detection
        html += '<div class="card mb-3">';
        html += '<div class="card-header"><h6 class="mb-0">Spam Analysis</h6></div>';
        html += '<div class="card-body">';
        const spamClass = data.spam_detection.risk_level === 'none' ? 'success' :
                         data.spam_detection.risk_level === 'low' ? 'info' : 'warning';
        html += '<span class="badge bg-' + spamClass + '">' + data.spam_detection.classification + '</span>';
        html += ' <span class="text-muted">(' + data.spam_detection.spam_score + '% spam probability)</span>';
        html += '</div></div>';

        // Credibility
        html += '<div class="card mb-3">';
        html += '<div class="card-header"><h6 class="mb-0">Credibility Assessment</h6></div>';
        html += '<div class="card-body">';
        const credClass = data.credibility.credibility_level === 'high' ? 'success' : 'info';
        html += '<span class="badge bg-' + credClass + '">' + data.credibility.classification + '</span>';
        html += ' <span class="text-muted">(' + data.credibility.credibility_score + '% credibility score)</span>';
        html += '</div></div>';

        resultDiv.innerHTML = html;
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('Professional analysis error');
    });
});

// Helper functions (use from advanced_nlp.js if available, otherwise define)
if (typeof createAdvancedResultsDiv === 'undefined') {
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
}

if (typeof showLoading === 'undefined') {
    function showLoading() {
        document.getElementById('loadingIndicator').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
    }
}

if (typeof hideLoading === 'undefined') {
    function hideLoading() {
        document.getElementById('loadingIndicator').style.display = 'none';
    }
}

if (typeof showResults === 'undefined') {
    function showResults() {
        document.getElementById('resultsSection').style.display = 'block';
    }
}
