import yfinance as yf
import pandas as pd
from datetime import datetime
import joblib
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockSentimentAnalyzer:
    def __init__(self, vectorizer_path="vectoriser.pkl", model_path="xgb_model.pkl"):
        """Initialize the analyzer with saved model and vectorizer."""
        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
            self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
            logger.info("Model and vectorizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or vectorizer: {str(e)}")
            raise

    def fetch_news(self, ticker, start_date, end_date):
        """Fetch news for a given ticker between start and end dates."""
        try:
            # Validate inputs
            if not isinstance(ticker, str) or not ticker:
                raise ValueError("Ticker must be a non-empty string")
            
            # Convert dates to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            if start_date > end_date:
                raise ValueError("Start date must be before end date")
                
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.warning(f"No news found for {ticker}")
                return pd.DataFrame(columns=["Sentence"])
            
            # Convert to DataFrame
            news_df = pd.DataFrame(news)
            
            # Filter by date - handle cases where providerPublishTime might be missing
            if 'providerPublishTime' in news_df.columns:
                news_df['providerPublishTime'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
                mask = (news_df['providerPublishTime'] >= start_date) & (news_df['providerPublishTime'] <= end_date)
                filtered_df = news_df.loc[mask]
            else:
                logger.warning("No publish time found in news data, returning all news")
                filtered_df = news_df
            
            # Create the text to analyze using title and summary (if available)
            filtered_df["Sentence"] = filtered_df.apply(
                lambda row: f"{row.get('title', '')} {row.get('summary', '')}", axis=1
            )
            
            return filtered_df[["Sentence"]]
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            raise
    
    def analyze_sentiment(self, news_df):
        """Predict sentiment for news dataframe."""
        try:
            if news_df.empty:
                logger.warning("Empty news dataframe, returning empty results")
                return pd.DataFrame(columns=["Sentence", "Sentiment"])
                
            # Transform the text
            X_transformed = self.vectorizer.transform(news_df["Sentence"])
            
            # Predict sentiments
            preds = self.model.predict(X_transformed)
            news_df["Sentiment"] = [self.label_map.get(p, "unknown") for p in preds]
            
            return news_df
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise
    
    def get_sentiment_distribution(self, sentiment_df):
        """Get the distribution of sentiments."""
        if sentiment_df.empty:
            return {}
            
        return sentiment_df["Sentiment"].value_counts().to_dict()

def main():
    # Example usage
    try:
        ticker = "AAPL"
        start_date = "2023-10-01"
        end_date = "2023-10-10"
        
        analyzer = StockSentimentAnalyzer()
        
        news_df = analyzer.fetch_news(ticker, start_date, end_date)
        results_df = analyzer.analyze_sentiment(news_df)
        
        # Print results
        print(f"Sentiment Analysis for {ticker} from {start_date} to {end_date}:")
        print(results_df[["Sentence", "Sentiment"]].head())
        
        # Get sentiment distribution
        distribution = analyzer.get_sentiment_distribution(results_df)
        print("\nSentiment Distribution:")
        for sentiment, count in distribution.items():
            print(f"{sentiment}: {count}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())