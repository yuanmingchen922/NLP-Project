# Yelp Review Analysis System

A comprehensive Natural Language Processing system for analyzing Yelp business reviews with 11+ advanced NLP features.

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-Academic-orange)

## 🚀 Features

### Core NLP Features
- ✅ **Sentiment Analysis** - Traditional ML and Deep Learning approaches
- ✅ **Named Entity Recognition** - Extract people, locations, products
- ✅ **Keyword Extraction** - TF-IDF + noun phrase extraction
- ✅ **Text Summarization** - Extractive summarization
- ✅ **Aspect-Based Sentiment** - Analyze 5 key aspects (food, service, atmosphere, price, cleanliness)
- ✅ **Dish Extraction** - Multi-strategy menu item detection
- ✅ **Review Quality Scoring** - 5-dimensional quality assessment
- ✅ **Spam Detection** - Identify promotional/fake reviews
- ✅ **Credibility Analysis** - Trust score calculation
- ✅ **Business Analytics** - Trend analysis and improvement scoring
- ✅ **Intelligent Search** - Advanced business search and filtering

### Technical Highlights
- 🔥 **100K reviews** from Yelp Academic Dataset
- 🎯 **85%+ accuracy** on sentiment classification
- ⚡ **<100ms response** time with caching
- 📊 **Real-time analytics** with interactive visualizations
- 🌐 **RESTful API** with 30+ endpoints
- 💾 **Smart caching** (1-hour TTL, 70%+ hit rate)

## 📊 Demo

### Search Interface
```
┌────────────────────────────────────────────┐
│  🔍 Search: "pizza restaurant"            │
└────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────┐
│ Mario's Pizza ⭐ 4.5 (245 reviews)        │
│ 123 Main St, New York                      │
│                                            │
│ [View Reviews] [Popular Dishes]            │
│                                            │
│ 📍 View on Maps  🔗 Find Website          │
└────────────────────────────────────────────┘
```

### API Example
```python
# Analyze sentiment
POST /api/analyze/sentiment
{
  "text": "Amazing food and great service!"
}
→ {"sentiment": "positive", "score": 0.92}

# Extract dishes
GET /api/business/abc123/dishes
→ {
    "dishes": [
      {"name": "Margherita Pizza", "rating": 4.8, "mentions": 45}
    ]
  }
```

## 🛠️ Tech Stack

### NLP & ML
- **spaCy** - NER, POS tagging, noun chunks
- **NLTK** - Text preprocessing, tokenization
- **Scikit-learn** - Logistic Regression, SVM, TF-IDF
- **Transformers** - BERT (optional)
- **TextBlob** - Sentiment analysis

### Backend
- **Flask** - REST API server
- **Pandas** - Data processing
- **NumPy/SciPy** - Statistical analysis

### Frontend
- **Bootstrap 5** - UI framework
- **Chart.js** - Data visualization
- **Vanilla JavaScript** - Interactivity

### External Services
- **OpenAI GPT-4o-mini** - AI-assisted dish extraction
- **SerpAPI** - (Optional) Real-time data fetching

## 📦 Installation

```bash
# 1. Clone repository
git clone https://github.com/yuanmingchen922/NLP-Project.git
cd NLP-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. (Optional) Set up API keys
cp .env.example .env
# Edit .env with your OpenAI/SerpAPI keys
```

## 🚀 Quick Start

```bash
# Start the server
python run.py

# Access web interface
open http://127.0.0.1:5001
```

The system will:
1. Load 100K preprocessed reviews
2. Initialize NLP models
3. Start Flask server on port 5001
4. Enable caching for fast responses

## 📖 Documentation

For detailed documentation, see **[NLP_FEATURES_DOCUMENTATION.md](NLP_FEATURES_DOCUMENTATION.md)**

### Key Documentation Sections
- **System Architecture** - Component overview and data flow
- **Complete Workflow** - Data pipeline from ingestion to deployment
- **NLP Features** - Detailed design process for each feature
- **API Reference** - All 30+ endpoints with examples
- **Usage Guide** - Installation, training, and usage instructions
- **Performance** - Optimization strategies and metrics

## 🎯 Use Cases

### For Businesses
- 📈 Monitor customer sentiment trends over time
- 🍕 Identify most popular menu items
- ⚠️ Detect quality issues early
- 📊 Compare performance with competitors

### For Customers
- 🔍 Search businesses by specific criteria
- 📝 Read AI-generated review summaries
- ⭐ See aspect-based ratings (food, service, etc.)
- 🎯 Discover top-rated dishes

### For Researchers
- 🧪 Experiment with NLP techniques
- 📊 Analyze large-scale review datasets
- 🤖 Train custom sentiment models
- 📈 Study customer behavior patterns

## 📊 Dataset

- **Source:** Yelp Academic Dataset
- **Total Reviews:** 6.9M (100K sampled)
- **Businesses:** 9,973 unique
- **Categories:** Restaurants, shops, services
- **Time Range:** Multiple years of historical data

## 🎨 Screenshots

### Main Search Interface
Clean, intuitive search with real-time results

### Business Details
Comprehensive analytics including:
- Review sentiment distribution
- Popular dishes with ratings
- Quality trends over time
- Aspect-based analysis

### Review Analysis
Individual review analysis showing:
- Sentiment score
- Quality rating
- Key entities
- Spam/credibility check

## 🔬 Model Performance

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| Sentiment (Logistic) | 87% | 0.87 | <50ms |
| NER (spaCy) | ~90% | 0.89 | <30ms |
| Quality Scoring | N/A | N/A | <100ms |
| Spam Detection | ~82% | 0.80 | <40ms |

## 🛣️ Roadmap

### Version 1.2 (Planned)
- [ ] Redis persistent caching
- [ ] Real-time WebSocket updates
- [ ] Multi-business comparison view
- [ ] PDF/Excel report export
- [ ] Mobile-responsive redesign

### Version 2.0 (Future)
- [ ] Image analysis (dish photos)
- [ ] Multi-language support
- [ ] Recommendation engine
- [ ] Aspect-based search
- [ ] Advanced visualizations

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for academic purposes. Please cite the Yelp Academic Dataset if using this code.

## 🙏 Acknowledgments

- **Yelp** - For providing the Academic Dataset
- **spaCy** - Excellent NLP toolkit
- **OpenAI** - GPT-4o-mini API
- **Bootstrap** - UI components
- **Chart.js** - Beautiful visualizations

## 📧 Contact

**Developer:** Mingchen Yuan
**GitHub:** [@yuanmingchen922](https://github.com/yuanmingchen922)
**Project Link:** [https://github.com/yuanmingchen922/NLP-Project](https://github.com/yuanmingchen922/NLP-Project)

---

⭐ If you find this project useful, please give it a star!

**Version:** 1.1.0
**Last Updated:** December 2, 2025
