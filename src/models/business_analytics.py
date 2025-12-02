"""
Business Intelligence and Analytics Module
Provides trend analysis, competitive insights, and business metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SentimentTrendAnalyzer:
    """
    Analyzes sentiment trends over time and identifies patterns
    """

    def __init__(self, reviews_data: pd.DataFrame = None):
        self.reviews_data = reviews_data

    def analyze_temporal_trends(self, business_id: str = None,
                                time_period: str = 'month') -> Dict[str, Any]:
        """
        Analyze sentiment trends over time
        time_period: 'day', 'week', 'month', 'quarter', 'year'
        """
        if self.reviews_data is None or len(self.reviews_data) == 0:
            return {"error": "No data available"}

        df = self.reviews_data.copy()

        # Filter by business if specified
        if business_id:
            df = df[df['business_id'] == business_id]

        if len(df) == 0:
            return {"error": "No reviews found"}

        # Ensure date column exists
        if 'date' not in df.columns:
            return {"error": "Date column not found"}

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Group by time period
        if time_period == 'day':
            df['period'] = df['date'].dt.date
        elif time_period == 'week':
            df['period'] = df['date'].dt.to_period('W')
        elif time_period == 'month':
            df['period'] = df['date'].dt.to_period('M')
        elif time_period == 'quarter':
            df['period'] = df['date'].dt.to_period('Q')
        else:  # year
            df['period'] = df['date'].dt.to_period('Y')

        # Calculate metrics per period
        trend_data = df.groupby('period').agg({
            'stars': ['mean', 'count', 'std'],
            'business_id': 'count'
        }).reset_index()

        trend_data.columns = ['period', 'avg_rating', 'review_count', 'rating_std', 'total_reviews']

        # Convert period to string for JSON serialization
        trend_data['period'] = trend_data['period'].astype(str)

        # Calculate trend direction
        if len(trend_data) >= 3:
            recent_avg = trend_data['avg_rating'].tail(3).mean()
            older_avg = trend_data['avg_rating'].head(3).mean()
            trend_direction = "improving" if recent_avg > older_avg else "declining"
            trend_strength = abs(recent_avg - older_avg)
        else:
            trend_direction = "stable"
            trend_strength = 0

        return {
            "trend_data": trend_data.to_dict('records'),
            "trend_direction": trend_direction,
            "trend_strength": round(trend_strength, 2),
            "summary": {
                "total_periods": len(trend_data),
                "avg_rating_overall": round(df['stars'].mean(), 2),
                "total_reviews": int(df.shape[0]),
                "rating_volatility": round(df['stars'].std(), 2)
            }
        }

    def identify_sentiment_shifts(self, business_id: str = None,
                                  threshold: float = 0.5) -> List[Dict]:
        """
        Identify significant sentiment shifts/events
        """
        if self.reviews_data is None or len(self.reviews_data) == 0:
            return []

        df = self.reviews_data.copy()

        if business_id:
            df = df[df['business_id'] == business_id]

        if len(df) < 10:
            return []

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

        # Calculate rolling average
        window = min(10, len(df) // 3)
        df['rolling_avg'] = df['stars'].rolling(window=window, min_periods=1).mean()

        # Find significant shifts
        shifts = []
        for i in range(window, len(df)):
            current_avg = df['rolling_avg'].iloc[i]
            previous_avg = df['rolling_avg'].iloc[i-window]

            change = current_avg - previous_avg

            if abs(change) >= threshold:
                shifts.append({
                    "date": str(df['date'].iloc[i].date()),
                    "change": round(change, 2),
                    "direction": "positive" if change > 0 else "negative",
                    "previous_rating": round(previous_avg, 2),
                    "new_rating": round(current_avg, 2),
                    "magnitude": "significant" if abs(change) >= 1.0 else "moderate"
                })

        return shifts

    def analyze_improvement_trends(self, business_id: str = None,
                                   period: str = 'quarter') -> Dict[str, Any]:
        """
        Analyze improvement trends with 0-100 improvement score (Version 1.1)

        Args:
            business_id: Business ID to analyze (None for all)
            period: Time period ('month', 'quarter', 'year')

        Returns:
            Dict with trend analysis and improvement metrics
        """
        import numpy as np
        from scipy import stats

        if self.reviews_data is None:
            return {}

        df = self.reviews_data.copy()

        if business_id:
            df = df[df['business_id'] == business_id]

        if len(df) < 5:  # Need minimum data for trend analysis
            return {
                'error': 'Insufficient data for trend analysis',
                'min_required': 5,
                'available': len(df)
            }

        # Parse dates and sort
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

        # Determine time period grouping
        period_map = {
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }
        freq = period_map.get(period, 'Q')

        # Group by period
        df['period'] = df['date'].dt.to_period(freq)
        grouped = df.groupby('period').agg({
            'stars': ['mean', 'count']
        }).reset_index()

        grouped.columns = ['period', 'avg_rating', 'review_count']
        grouped = grouped[grouped['review_count'] >= 2]  # Filter periods with few reviews

        if len(grouped) < 3:
            return {
                'error': 'Insufficient periods for trend analysis',
                'min_required': 3,
                'available': len(grouped)
            }

        # Convert period to numeric for regression
        grouped['period_num'] = range(len(grouped))

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            grouped['period_num'], grouped['avg_rating']
        )

        # Calculate improvement score (0-100)
        improvement_score = self._calculate_improvement_score(
            slope, grouped['avg_rating'].values, r_value
        )

        # Classify trend direction
        if slope > 0.1 and improvement_score > 70:
            trend_direction = 'Strongly Improving'
            trend_class = 'success'
        elif slope > 0.05:
            trend_direction = 'Improving'
            trend_class = 'success'
        elif abs(slope) <= 0.05:
            trend_direction = 'Stable'
            trend_class = 'info'
        elif slope < -0.1:
            trend_direction = 'Declining'
            trend_class = 'danger'
        else:
            trend_direction = 'Slightly Declining'
            trend_class = 'warning'

        # Prepare chart data
        chart_data = []
        for _, row in grouped.iterrows():
            chart_data.append({
                'period': str(row['period']),
                'average_rating': round(row['avg_rating'], 2),
                'review_count': int(row['review_count'])
            })

        # Calculate predicted next period rating
        next_period_num = len(grouped)
        predicted_rating = slope * next_period_num + intercept
        predicted_rating = max(1.0, min(5.0, predicted_rating))  # Clamp to 1-5

        return {
            'improvement_score': round(improvement_score, 1),
            'trend_direction': trend_direction,
            'trend_class': trend_class,
            'slope': round(slope, 4),
            'correlation': round(r_value, 3),
            'p_value': round(p_value, 4),
            'statistical_significance': 'significant' if p_value < 0.05 else 'not_significant',
            'current_rating': round(grouped['avg_rating'].iloc[-1], 2),
            'first_period_rating': round(grouped['avg_rating'].iloc[0], 2),
            'predicted_next_rating': round(predicted_rating, 2),
            'total_change': round(grouped['avg_rating'].iloc[-1] - grouped['avg_rating'].iloc[0], 2),
            'period_count': len(grouped),
            'total_reviews': int(grouped['review_count'].sum()),
            'chart_data': chart_data
        }

    def _calculate_improvement_score(self, slope: float, ratings: np.ndarray,
                                     correlation: float) -> float:
        """
        Calculate improvement score (0-100) based on multiple components

        Args:
            slope: Linear regression slope
            ratings: Array of period average ratings
            correlation: Correlation coefficient (R)

        Returns:
            Improvement score (0-100)
        """
        import numpy as np

        # Component 1: Trend direction (40 points max)
        # Positive slope = points, negative = penalty
        trend_score = min(40, max(-40, slope * 100))

        # Component 2: Performance improvement (40 points max)
        # Based on how much rating improved from start to end
        if len(ratings) >= 2:
            start_rating = ratings[0]
            end_rating = ratings[-1]
            change = end_rating - start_rating

            # Scale: 0 change = 20 points, +1 star = 40 points, -1 star = 0 points
            performance_score = 20 + (change * 20)
            performance_score = min(40, max(0, performance_score))
        else:
            performance_score = 20

        # Component 3: Consistency (20 points max)
        # Higher correlation = more consistent trend
        consistency_score = abs(correlation) * 20

        # Total score
        total_score = trend_score + performance_score + consistency_score

        # Normalize to 0-100
        # Theoretical range: -40 to 100, shift and scale to 0-100
        normalized_score = ((total_score + 40) / 140) * 100

        return max(0, min(100, normalized_score))


class BusinessInsightsAnalyzer:
    """
    Provides business intelligence insights from review data
    """

    def __init__(self, reviews_data: pd.DataFrame = None,
                 businesses_data: pd.DataFrame = None):
        self.reviews_data = reviews_data
        self.businesses_data = businesses_data

    def generate_business_scorecard(self, business_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive business scorecard with key metrics
        """
        if self.reviews_data is None:
            return {"error": "No review data available"}

        business_reviews = self.reviews_data[
            self.reviews_data['business_id'] == business_id
        ]

        if len(business_reviews) == 0:
            return {"error": "No reviews found for this business"}

        # Calculate key metrics
        total_reviews = len(business_reviews)
        avg_rating = business_reviews['stars'].mean()
        rating_distribution = business_reviews['stars'].value_counts().to_dict()

        # Calculate rating trends
        business_reviews['date'] = pd.to_datetime(business_reviews['date'],
                                                  errors='coerce')
        recent_reviews = business_reviews[
            business_reviews['date'] >= (datetime.now() - timedelta(days=90))
        ]

        recent_avg = recent_reviews['stars'].mean() if len(recent_reviews) > 0 else avg_rating
        trend = "improving" if recent_avg > avg_rating else "stable" if recent_avg == avg_rating else "declining"

        # Sentiment distribution
        def categorize_sentiment(rating):
            if rating >= 4:
                return "positive"
            elif rating >= 3:
                return "neutral"
            else:
                return "negative"

        sentiment_dist = business_reviews['stars'].apply(categorize_sentiment).value_counts().to_dict()

        # Review velocity (reviews per month)
        if len(business_reviews) > 0 and business_reviews['date'].notna().any():
            date_range = (business_reviews['date'].max() - business_reviews['date'].min()).days
            months = max(1, date_range / 30)
            review_velocity = total_reviews / months
        else:
            review_velocity = 0

        return {
            "business_id": business_id,
            "metrics": {
                "total_reviews": total_reviews,
                "average_rating": round(avg_rating, 2),
                "recent_average_rating": round(recent_avg, 2),
                "rating_distribution": rating_distribution,
                "sentiment_distribution": sentiment_dist,
                "review_velocity": round(review_velocity, 2),
                "trend": trend
            },
            "insights": self._generate_scorecard_insights(
                avg_rating, recent_avg, total_reviews, review_velocity, trend
            )
        }

    def compare_competitors(self, business_ids: List[str],
                           metric: str = 'rating') -> Dict[str, Any]:
        """
        Compare multiple businesses on specified metrics
        """
        if self.reviews_data is None:
            return {"error": "No data available"}

        comparison_data = []

        for business_id in business_ids:
            business_reviews = self.reviews_data[
                self.reviews_data['business_id'] == business_id
            ]

            if len(business_reviews) == 0:
                continue

            data = {
                "business_id": business_id,
                "avg_rating": round(business_reviews['stars'].mean(), 2),
                "total_reviews": len(business_reviews),
                "rating_std": round(business_reviews['stars'].std(), 2)
            }

            # Get business name if available
            if self.businesses_data is not None:
                business_info = self.businesses_data[
                    self.businesses_data['business_id'] == business_id
                ]
                if len(business_info) > 0:
                    data['business_name'] = business_info['name'].iloc[0]

            comparison_data.append(data)

        # Rank businesses
        if metric == 'rating':
            comparison_data.sort(key=lambda x: x['avg_rating'], reverse=True)
        elif metric == 'reviews':
            comparison_data.sort(key=lambda x: x['total_reviews'], reverse=True)

        return {
            "comparison": comparison_data,
            "leader": comparison_data[0] if comparison_data else None,
            "metric": metric
        }

    def analyze_aspect_importance(self, business_id: str = None) -> Dict[str, Any]:
        """
        Analyze which aspects are most important for customer satisfaction
        """
        if self.reviews_data is None:
            return {"error": "No data available"}

        df = self.reviews_data.copy()

        if business_id:
            df = df[df['business_id'] == business_id]

        if len(df) == 0:
            return {"error": "No reviews found"}

        # Aspect keywords
        aspects = {
            'food_quality': ['food', 'meal', 'dish', 'taste', 'delicious', 'flavor'],
            'service': ['service', 'staff', 'waiter', 'server', 'friendly', 'attentive'],
            'atmosphere': ['atmosphere', 'ambiance', 'decor', 'environment', 'cozy'],
            'value': ['price', 'value', 'expensive', 'worth', 'affordable'],
            'cleanliness': ['clean', 'hygiene', 'sanitary', 'tidy']
        }

        aspect_ratings = {}

        for aspect, keywords in aspects.items():
            # Find reviews mentioning this aspect
            if 'text' in df.columns:
                aspect_reviews = df[
                    df['text'].str.lower().str.contains('|'.join(keywords), na=False)
                ]

                if len(aspect_reviews) > 0:
                    aspect_ratings[aspect] = {
                        "avg_rating": round(aspect_reviews['stars'].mean(), 2),
                        "mention_count": len(aspect_reviews),
                        "mention_percentage": round(len(aspect_reviews) / len(df) * 100, 1)
                    }

        # Sort by importance (combination of rating and frequency)
        for aspect in aspect_ratings:
            aspect_ratings[aspect]['importance_score'] = (
                aspect_ratings[aspect]['avg_rating'] * 0.6 +
                aspect_ratings[aspect]['mention_percentage'] * 0.04
            )

        sorted_aspects = sorted(
            aspect_ratings.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )

        return {
            "aspects": dict(sorted_aspects),
            "top_priority": sorted_aspects[0][0] if sorted_aspects else None,
            "total_reviews_analyzed": len(df)
        }

    def _generate_scorecard_insights(self, avg_rating, recent_avg,
                                    total_reviews, velocity, trend) -> List[str]:
        """Generate insights for business scorecard"""
        insights = []

        if avg_rating >= 4.0:
            insights.append("Strong overall rating indicates high customer satisfaction")
        elif avg_rating < 3.0:
            insights.append("Below-average rating requires immediate attention")

        if trend == "improving":
            insights.append("Recent reviews show improvement in customer satisfaction")
        elif trend == "declining":
            insights.append("Recent trend shows declining satisfaction - investigate issues")

        if velocity > 10:
            insights.append("High review volume indicates strong customer engagement")
        elif velocity < 1:
            insights.append("Low review volume - encourage customers to leave feedback")

        if total_reviews < 10:
            insights.append("Limited review data - gather more customer feedback")

        return insights


def generate_executive_summary(reviews_data: pd.DataFrame,
                              business_id: str = None) -> Dict[str, Any]:
    """
    Generate executive summary with key business insights
    """
    trend_analyzer = SentimentTrendAnalyzer(reviews_data)
    insights_analyzer = BusinessInsightsAnalyzer(reviews_data)

    if business_id:
        scorecard = insights_analyzer.generate_business_scorecard(business_id)
        trends = trend_analyzer.analyze_temporal_trends(business_id, 'month')
        shifts = trend_analyzer.identify_sentiment_shifts(business_id)
        aspects = insights_analyzer.analyze_aspect_importance(business_id)
    else:
        # Overall summary
        scorecard = {
            "metrics": {
                "total_reviews": len(reviews_data),
                "average_rating": round(reviews_data['stars'].mean(), 2)
            }
        }
        trends = trend_analyzer.analyze_temporal_trends(time_period='month')
        shifts = []
        aspects = {}

    return {
        "scorecard": scorecard,
        "trends": trends,
        "significant_events": shifts[:5],  # Top 5 shifts
        "aspect_analysis": aspects,
        "generated_at": datetime.now().isoformat()
    }
