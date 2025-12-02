/**
 * Yelp评论分析系统 - 主JavaScript文件
 */

// 工具函数
const utils = {
    // 格式化数字
    formatNumber: function(num) {
        return num.toLocaleString();
    },

    // 格式化日期
    formatDate: function(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('zh-CN');
    },

    // 截断文本
    truncateText: function(text, maxLength = 100) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    },

    // 显示错误消息
    showError: function(message) {
        alert('错误: ' + message);
    },

    // 显示成功消息
    showSuccess: function(message) {
        alert('成功: ' + message);
    }
};

// API调用函数
const api = {
    baseUrl: '/api',

    // 获取商户列表
    getBusinesses: async function(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const response = await fetch(`${this.baseUrl}/businesses?${queryString}`);
        return await response.json();
    },

    // 获取商户详情
    getBusiness: async function(businessId) {
        const response = await fetch(`${this.baseUrl}/business/${businessId}`);
        return await response.json();
    },

    // 获取商户评论
    getBusinessReviews: async function(businessId, limit = 50) {
        const response = await fetch(`${this.baseUrl}/business/${businessId}/reviews?limit=${limit}`);
        return await response.json();
    },

    // 分析情感
    analyzeSentiment: async function(text) {
        const response = await fetch(`${this.baseUrl}/analyze/sentiment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        return await response.json();
    },

    // 分析文本统计
    analyzeStatistics: async function(text) {
        const response = await fetch(`${this.baseUrl}/analyze/statistics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        return await response.json();
    },

    // 获取统计概览
    getStatsOverview: async function() {
        const response = await fetch(`${this.baseUrl}/stats/overview`);
        return await response.json();
    },

    // 搜索评论
    searchReviews: async function(query, limit = 20) {
        const response = await fetch(`${this.baseUrl}/search/reviews?q=${encodeURIComponent(query)}&limit=${limit}`);
        return await response.json();
    }
};

// UI组件
const ui = {
    // 创建评分星星
    createStars: function(rating) {
        const fullStars = Math.floor(rating);
        const halfStar = rating % 1 >= 0.5 ? 1 : 0;
        const emptyStars = 5 - fullStars - halfStar;

        let stars = '';
        stars += '⭐'.repeat(fullStars);
        if (halfStar) stars += '✨';
        stars += '☆'.repeat(emptyStars);

        return stars;
    },

    // 创建进度条
    createProgressBar: function(value, max = 100) {
        const percentage = (value / max) * 100;
        return `
            <div class="progress">
                <div class="progress-bar" role="progressbar"
                    style="width: ${percentage}%"
                    aria-valuenow="${value}" aria-valuemin="0" aria-valuemax="${max}">
                    ${percentage.toFixed(1)}%
                </div>
            </div>
        `;
    },

    // 创建情感徽章
    createSentimentBadge: function(sentiment) {
        const badges = {
            'Positive': '<span class="badge bg-success">正面</span>',
            'Neutral': '<span class="badge bg-warning">中性</span>',
            'Negative': '<span class="badge bg-danger">负面</span>'
        };
        return badges[sentiment] || '<span class="badge bg-secondary">未知</span>';
    },

    // 创建加载动画
    createLoadingSpinner: function() {
        return `
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">加载中...</p>
            </div>
        `;
    }
};

// 图表工具
const charts = {
    // 创建评分分布图
    createRatingDistChart: function(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data).map(k => k + '星'),
                datasets: [{
                    label: '评论数量',
                    data: Object.values(data),
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(253, 126, 20, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(13, 202, 240, 0.8)',
                        'rgba(25, 135, 84, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    },

    // 创建饼图
    createPieChart: function(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(25, 135, 84, 0.8)',
                        'rgba(13, 110, 253, 0.8)',
                        'rgba(253, 126, 20, 0.8)'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    },

    // 创建折线图
    createLineChart: function(canvasId, labels, data, label = '数据') {
        const ctx = document.getElementById(canvasId);
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
};

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('Yelp Review System loaded successfully');

    // 可以添加全局初始化代码
});

// ============================================================================
// VERSION 1.1: ENHANCED FEATURES JAVASCRIPT
// ============================================================================

// Track active Chart.js instances to prevent memory leaks
const activeCharts = {};

// 1. Load and show aspect analysis
async function loadAndShowAspects(businessId) {
    const section = document.getElementById(`aspects-${businessId}`);
    const loading = document.getElementById(`aspects-loading-${businessId}`);
    const content = document.getElementById(`aspects-content-${businessId}`);

    // Show section and loading spinner
    section.style.display = 'block';
    loading.style.display = 'block';
    content.style.display = 'none';

    try {
        const response = await fetch(`/api/business/${businessId}/aspects-enhanced`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Hide loading, show content
        loading.style.display = 'none';
        content.style.display = 'block';

        // Render radar chart and scores
        renderAspectRadarChart(businessId, data);
        renderAspectScoresList(businessId, data);

    } catch (error) {
        console.error('Error loading aspects:', error);
        loading.innerHTML = `<p class="text-danger small">Error loading data: ${error.message}</p>`;
    }
}

// 2. Render aspect radar chart using Chart.js
function renderAspectRadarChart(businessId, data) {
    const canvasId = `radar-chart-${businessId}`;
    const ctx = document.getElementById(canvasId);

    // Destroy existing chart if any
    if (activeCharts[canvasId]) {
        activeCharts[canvasId].destroy();
    }

    // Create new radar chart
    activeCharts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.radar_chart_data.labels,
            datasets: [{
                label: 'Aspect Ratings',
                data: data.radar_chart_data.ratings,
                backgroundColor: 'rgba(220, 53, 69, 0.2)',
                borderColor: 'rgb(220, 53, 69)',
                borderWidth: 2,
                pointBackgroundColor: 'rgb(220, 53, 69)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(220, 53, 69)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// 3. Render aspect scores list with progress bars
function renderAspectScoresList(businessId, data) {
    const container = document.getElementById(`aspect-scores-${businessId}`);
    let html = '';

    for (const [key, aspect] of Object.entries(data.aspects)) {
        const percentage = (aspect.average_rating / 5) * 100;
        const color = aspect.average_rating >= 4 ? 'success' : aspect.average_rating >= 3 ? 'warning' : 'danger';

        html += `
            <div class="mb-2">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <small class="fw-bold">${aspect.label}</small>
                    <small class="text-muted">${aspect.average_rating.toFixed(1)}</small>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar bg-${color}" role="progressbar"
                         style="width: ${percentage}%"
                         aria-valuenow="${aspect.average_rating}"
                         aria-valuemin="0"
                         aria-valuemax="5"></div>
                </div>
                <small class="text-muted">${aspect.mention_count} mentions (${aspect.coverage_percentage}%)</small>
            </div>
        `;
    }

    container.innerHTML = html;
}

// 4. Toggle aspects section
function toggleAspects(businessId) {
    const section = document.getElementById(`aspects-${businessId}`);
    section.style.display = 'none';

    // Destroy chart to free memory
    const canvasId = `radar-chart-${businessId}`;
    if (activeCharts[canvasId]) {
        activeCharts[canvasId].destroy();
        delete activeCharts[canvasId];
    }
}

// 5. Load and show trends analysis
async function loadAndShowTrends(businessId) {
    const section = document.getElementById(`trends-${businessId}`);
    const loading = document.getElementById(`trends-loading-${businessId}`);
    const content = document.getElementById(`trends-content-${businessId}`);

    // Show section and loading spinner
    section.style.display = 'block';
    loading.style.display = 'block';
    content.style.display = 'none';

    try {
        const response = await fetch(`/api/business/${businessId}/improvement-trends?period=quarter`);
        const data = await response.json();

        if (data.error || data.trends.error) {
            throw new Error(data.trends?.error || data.error);
        }

        // Hide loading, show content
        loading.style.display = 'none';
        content.style.display = 'block';

        // Render improvement badge and trend chart
        renderImprovementBadge(businessId, data.trends);
        renderTrendChart(businessId, data.trends);

    } catch (error) {
        console.error('Error loading trends:', error);
        loading.innerHTML = `<p class="text-danger small">Error loading data: ${error.message}</p>`;
    }
}

// 6. Render trend line chart
function renderTrendChart(businessId, trends) {
    const canvasId = `trend-chart-${businessId}`;
    const ctx = document.getElementById(canvasId);

    // Destroy existing chart if any
    if (activeCharts[canvasId]) {
        activeCharts[canvasId].destroy();
    }

    const labels = trends.chart_data.map(d => d.period);
    const ratings = trends.chart_data.map(d => d.average_rating);

    // Create new line chart
    activeCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Rating',
                data: ratings,
                borderColor: trends.trend_class === 'success' ? 'rgb(40, 167, 69)' :
                           trends.trend_class === 'danger' ? 'rgb(220, 53, 69)' :
                           'rgb(23, 162, 184)',
                backgroundColor: trends.trend_class === 'success' ? 'rgba(40, 167, 69, 0.1)' :
                                trends.trend_class === 'danger' ? 'rgba(220, 53, 69, 0.1)' :
                                'rgba(23, 162, 184, 0.1)',
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 1,
                    max: 5,
                    ticks: {
                        stepSize: 0.5
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// 7. Render improvement badge
function renderImprovementBadge(businessId, trends) {
    const container = document.getElementById(`improvement-badge-${businessId}`);

    const badgeClass = trends.trend_class === 'success' ? 'bg-success' :
                      trends.trend_class === 'danger' ? 'bg-danger' :
                      trends.trend_class === 'warning' ? 'bg-warning' :
                      'bg-info';

    container.innerHTML = `
        <div class="card border-${trends.trend_class}">
            <div class="card-body p-3">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-1">Improvement Score</h6>
                        <h2 class="mb-0 text-${trends.trend_class}">${trends.improvement_score}/100</h2>
                        <span class="badge ${badgeClass} mt-2">${trends.trend_direction}</span>
                    </div>
                    <div class="text-end">
                        <small class="text-muted d-block">Total Change</small>
                        <strong class="${trends.total_change >= 0 ? 'text-success' : 'text-danger'}">
                            ${trends.total_change >= 0 ? '+' : ''}${trends.total_change.toFixed(2)}
                        </strong>
                        <small class="text-muted d-block mt-2">
                            ${trends.first_period_rating.toFixed(1)} → ${trends.current_rating.toFixed(1)}
                        </small>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 8. Toggle trends section
function toggleTrends(businessId) {
    const section = document.getElementById(`trends-${businessId}`);
    section.style.display = 'none';

    // Destroy chart to free memory
    const canvasId = `trend-chart-${businessId}`;
    if (activeCharts[canvasId]) {
        activeCharts[canvasId].destroy();
        delete activeCharts[canvasId];
    }
}

// 9. Load and show popular dishes
async function loadAndShowDishes(businessId) {
    const section = document.getElementById(`dishes-${businessId}`);
    const loading = document.getElementById(`dishes-loading-${businessId}`);
    const content = document.getElementById(`dishes-content-${businessId}`);

    // Show section and loading spinner
    section.style.display = 'block';
    loading.style.display = 'block';
    content.style.display = 'none';

    try {
        const response = await fetch(`/api/business/${businessId}/dishes?top_n=10`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Hide loading, show content
        loading.style.display = 'none';
        content.style.display = 'block';

        // Render dishes list
        if (data.dishes && data.dishes.length > 0) {
            let html = '<div class="list-group list-group-flush">';

            data.dishes.forEach((dish, index) => {
                const sentimentBadge = dish.sentiment === 'positive' ?
                    '<span class="badge bg-success">Positive</span>' :
                    dish.sentiment === 'negative' ?
                    '<span class="badge bg-danger">Negative</span>' :
                    '<span class="badge bg-secondary">Neutral</span>';

                html += `
                    <div class="list-group-item px-0">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">${index + 1}. ${dish.name}</h6>
                                <small class="text-muted">
                                    ${dish.mention_count} mention${dish.mention_count > 1 ? 's' : ''} •
                                    ${dish.average_rating.toFixed(1)} average
                                </small>
                            </div>
                            <div>
                                ${sentimentBadge}
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            content.innerHTML = html;
        } else {
            content.innerHTML = '<p class="text-muted small">No dishes found in reviews.</p>';
        }

    } catch (error) {
        console.error('Error loading dishes:', error);
        loading.innerHTML = `<p class="text-danger small">Error loading data: ${error.message}</p>`;
    }
}

// 10. Toggle dishes section
function toggleDishes(businessId) {
    const section = document.getElementById(`dishes-${businessId}`);
    section.style.display = 'none';
}

// Clean up charts on page unload to prevent memory leaks
window.addEventListener('beforeunload', function() {
    for (const chartId in activeCharts) {
        if (activeCharts[chartId]) {
            activeCharts[chartId].destroy();
        }
    }
});

// 导出到全局作用域
window.YelpApp = {
    utils: utils,
    api: api,
    ui: ui,
    charts: charts
};
