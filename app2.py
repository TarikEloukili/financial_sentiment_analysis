import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
import altair as alt

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
            st.error(f"Failed to load model or vectorizer: {str(e)}")
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
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            st.error(f"Error fetching news for {ticker}: {str(e)}")
            return pd.DataFrame(columns=["Sentence"])
    
    def analyze_sentiment(self, news_df):
        """Predict sentiment for news dataframe."""
        try:
            if news_df.empty or "Sentence" not in news_df.columns:
                logger.warning("Empty news dataframe or missing 'Sentence' column, returning empty results")
                return pd.DataFrame(columns=["Sentence", "Sentiment"])
                
            # Transform the text
            X_transformed = self.vectorizer.transform(news_df["Sentence"])
            
            # Predict sentiments
            preds = self.model.predict(X_transformed)
            news_df["Sentiment"] = [self.label_map.get(p, "unknown") for p in preds]
            
            # Add numeric sentiment for plotting
            news_df["SentimentValue"] = news_df["Sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})
            
            return news_df
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            st.error(f"Error analyzing sentiment: {str(e)}")
            return pd.DataFrame(columns=["Sentence", "Sentiment", "SentimentValue"])
    
    def get_sentiment_distribution(self, sentiment_df):
        """Get the distribution of sentiments."""
        if sentiment_df.empty or "Sentiment" not in sentiment_df.columns:
            return {}
            
        return sentiment_df["Sentiment"].value_counts().to_dict()
        
    def get_investment_recommendation(self, sentiment_df):
        """Generate investment recommendation based on sentiment analysis."""
        if sentiment_df.empty or "Sentiment" not in sentiment_df.columns:
            return "Insufficient data to make a recommendation."
            
        sentiment_counts = sentiment_df["Sentiment"].value_counts()
        total = sentiment_counts.sum()
        
        if total == 0:
            return "Insufficient data to make a recommendation."
            
        negative_percentage = (sentiment_counts.get("negative", 0) / total) * 100
        positive_percentage = (sentiment_counts.get("positive", 0) / total) * 100
        
        if negative_percentage >= 70:
            return "⚠️ WARNING: It is not recommended to invest in this share. The sentiment is predominantly negative."
        elif positive_percentage >= 70:
            return "✅ RECOMMENDATION: This could be a good investment opportunity. The sentiment is predominantly positive."
        else:
            return "⚖️ NEUTRAL: The sentiment is mixed. Consider further research before investing."

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock price data for the given ticker and date range."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_popular_tickers():
    """Return a list of popular stock tickers."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "JPM", "BAC", "WMT",
        "DIS", "NFLX", "INTC", "AMD", "CSCO",
        "PFE", "JNJ", "KO", "PEP", "MCD"
    ]

# Create the Streamlit app
def main():
    st.set_page_config(
        page_title="Stock Sentiment Analysis", 
        page_icon="📈", 
        layout="wide"
    )
    
    st.title("📊 Stock Sentiment Analysis")
    st.write("Analyze sentiment from news articles to guide investment decisions")
    
    # Check if model files exist
    if not (os.path.exists("vectoriser.pkl") and os.path.exists("xgb_model.pkl")):
        st.error("⚠️ Model files not found. Please make sure 'vectoriser.pkl' and 'xgb_model.pkl' are in the current directory.")
        st.info("Note: This app requires pre-trained sentiment analysis models to function.")
        return
    
    # Initialize analyzer
    try:
        analyzer = StockSentimentAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize the analyzer: {str(e)}")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📝 Input Parameters")
    
    # Ticker selection
    popular_tickers = get_popular_tickers()
    ticker_input = st.sidebar.selectbox(
        "Select a company ticker:", 
        options=popular_tickers,
        index=0
    )
    
    custom_ticker = st.sidebar.text_input("Or enter a custom ticker:")
    ticker = custom_ticker if custom_ticker else ticker_input
    
    # Date range selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)  # Default to last 30 days
    
    start_date = st.sidebar.date_input(
        "Start date:",
        value=start_date,
        max_value=end_date
    )
    
    end_date = st.sidebar.date_input(
        "End date:",
        value=end_date,
        min_value=start_date
    )
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This app analyzes news sentiment for publicly traded companies "
        "and provides investment recommendations based on sentiment analysis. "
        "Data is fetched from Yahoo Finance API."
    )
    
    # Run analysis button
    analyze_button = st.sidebar.button("🔍 Analyze Sentiment", type="primary")
    
    if analyze_button:
        with st.spinner(f"Fetching news for {ticker}..."):
            # Show company info
            try:
                company_info = yf.Ticker(ticker).info
                company_name = company_info.get('longName', ticker)
                st.header(f"Analysis for: {company_name} ({ticker})")
                
                # Display company details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${company_info.get('currentPrice', 'N/A')}")
                with col2:
                    st.metric("Market Cap", f"${company_info.get('marketCap', 'N/A'):,}" if 'marketCap' in company_info else "N/A")
                with col3:
                    st.metric("52-Week High", f"${company_info.get('fiftyTwoWeekHigh', 'N/A')}")
            except:
                st.header(f"Analysis for: {ticker}")
            
            # Fetch news and analyze sentiment
            news_df = analyzer.fetch_news(ticker, start_date, end_date)
            
            if news_df.empty:
                st.warning(f"No news found for {ticker} in the selected date range")
            else:
                sentiment_df = analyzer.analyze_sentiment(news_df)
                
                if sentiment_df.empty:
                    st.warning("Could not analyze sentiment from the news data")
                else:
                    # Show results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "📰 News", "📈 Stock Performance"])
                    
                    with tab1:
                        st.subheader("Sentiment Analysis Results")
                        
                        # Get sentiment counts
                        sentiment_counts = sentiment_df["Sentiment"].value_counts().reset_index()
                        sentiment_counts.columns = ["Sentiment", "Count"]
                        
                        # Sort sentiments in proper order
                        sentiment_order = ["positive", "neutral", "negative"]
                        sentiment_counts["Sentiment"] = pd.Categorical(
                            sentiment_counts["Sentiment"], 
                            categories=sentiment_order, 
                            ordered=True
                        )
                        sentiment_counts = sentiment_counts.sort_values("Sentiment")
                        
                        # Create two columns layout
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Create bar chart with Altair
                            colors = {'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}
                            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                                x=alt.X('Sentiment', sort=sentiment_order),
                                y='Count',
                                color=alt.Color('Sentiment', scale=alt.Scale(
                                    domain=list(colors.keys()),
                                    range=list(colors.values())
                                )),
                                tooltip=['Sentiment', 'Count']
                            ).properties(
                                title=f'Sentiment Distribution for {ticker}'
                            ).configure_axis(
                                labelFontSize=12,
                                titleFontSize=14
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                        
                        with col2:
                            # Display metrics
                            total_news = len(sentiment_df)
                            st.metric("Total News Articles", total_news)
                            
                            for sentiment in sentiment_order:
                                count = sentiment_counts[sentiment_counts["Sentiment"] == sentiment]["Count"].values
                                percentage = (count[0] / total_news * 100) if len(count) > 0 else 0
                                st.metric(
                                    f"{sentiment.capitalize()} Sentiment", 
                                    f"{count[0] if len(count) > 0 else 0} ({percentage:.1f}%)"
                                )
                        
                        # Show recommendation
                        recommendation = analyzer.get_investment_recommendation(sentiment_df)
                        st.markdown(f"### Investment Recommendation\n{recommendation}")
                        
                    with tab2:
                        st.subheader("News Articles")
                        
                        # Display news with sentiment
                        if news_df.empty:
                            st.info("No news found for the selected date range.")
                        else:
                            # Debug information to see what columns are available
                            st.write("Available columns:", list(news_df.columns))
                            
                            # Create a more robust display of news
                            news_display = pd.DataFrame()
                            
                            # Try to get the title or headline
                            if 'title' in news_df.columns:
                                news_display['Headline'] = news_df['title']
                            elif 'headline' in news_df.columns:
                                news_display['Headline'] = news_df['headline']
                            else:
                                # Create a generic title if none available
                                news_display['Headline'] = ["News Article " + str(i+1) for i in range(len(news_df))]
                            
                            # Add the sentiment if available
                            if 'Sentiment' in sentiment_df.columns:
                                news_display['Sentiment'] = sentiment_df['Sentiment']
                                
                                # Format the sentiment with colors
                                def format_sentiment(sentiment):
                                    if sentiment == 'positive':
                                        return f'<span style="color: green; font-weight: bold;">{sentiment.capitalize()}</span>'
                                    elif sentiment == 'negative':
                                        return f'<span style="color: red; font-weight: bold;">{sentiment.capitalize()}</span>'
                                    else:
                                        return f'<span style="color: orange; font-weight: bold;">{sentiment.capitalize()}</span>'
                                    
                                news_display['Sentiment'] = news_display['Sentiment'].apply(format_sentiment)
                            
                            # Add date if available
                            if 'providerPublishTime' in news_df.columns:
                                news_display['Date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s').dt.strftime('%Y-%m-%d')
                            
                            # Add summary if available
                            if 'summary' in news_df.columns:
                                news_display['Summary'] = news_df['summary'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
                            
                            # Display the news
                            st.write(news_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                            
                            # Also provide the raw news data option for debugging
                            with st.expander("Debug: View Raw News Data"):
                                st.dataframe(news_df)
                    
                    with tab3:
                        st.subheader("Stock Performance")
                        
                        # Fetch stock data
                        stock_data = fetch_stock_data(ticker, start_date, end_date)
                        
                        if not stock_data.empty:
                            # Use Streamlit's native plotting capabilities instead
                            st.subheader("Price Chart")
                            st.line_chart(stock_data['Close'])
                            
                            st.subheader("Volume Chart")
                            st.bar_chart(stock_data['Volume'])
                            
                            # Show key statistics
                            st.subheader("Key Statistics")
                            col1, col2, col3 = st.columns(3)

                            try:
                                with col1:
                                    if not stock_data.empty and len(stock_data) > 1:
                                        price_change = float(stock_data['Close'].iloc[-1]) - float(stock_data['Close'].iloc[0])
                                        price_change_pct = (price_change / float(stock_data['Close'].iloc[0])) * 100 if float(stock_data['Close'].iloc[0]) != 0 else 0
                                        
                                        # Format the values safely with error handling
                                        try:
                                            price_change_str = f"${price_change:.2f}"
                                            price_change_pct_str = f"{price_change_pct:.2f}%"
                                        except:
                                            price_change_str = "N/A"
                                            price_change_pct_str = "N/A"
                                            
                                        st.metric(
                                            "Price Change", 
                                            price_change_str,
                                            price_change_pct_str
                                        )
                                    else:
                                        st.metric("Price Change", "N/A")
                            except Exception as e:
                                st.metric("Price Change", "Error calculating")
                                st.error(f"Error with price change calculation: {str(e)}")

                            try:
                                with col2:
                                    if not stock_data.empty and 'Volume' in stock_data.columns:
                                        avg_volume = stock_data['Volume'].mean()
                                        if not pd.isna(avg_volume):
                                            st.metric("Average Volume", f"{int(avg_volume):,}")
                                        else:
                                            st.metric("Average Volume", "N/A")
                                    else:
                                        st.metric("Average Volume", "N/A")
                            except Exception as e:
                                st.metric("Average Volume", "Error calculating")

                            try:
                                with col3:
                                    if not stock_data.empty and 'Close' in stock_data.columns:
                                        volatility = stock_data['Close'].std()
                                        if not pd.isna(volatility):
                                            st.metric("Volatility", f"{volatility:.2f}")
                                        else:
                                            st.metric("Volatility", "N/A")
                                    else:
                                        st.metric("Volatility", "N/A")
                            except Exception as e:
                                st.metric("Volatility", "Error calculating")

if __name__ == "__main__":
    main()