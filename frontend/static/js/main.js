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

// 导出到全局作用域
window.YelpApp = {
    utils: utils,
    api: api,
    ui: ui,
    charts: charts
};
