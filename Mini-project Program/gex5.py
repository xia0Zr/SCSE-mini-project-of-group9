from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalysis:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def load_data(self, data):
        """Load the dataset to be analyzed for sentiment."""
        self.data = data

    def vader_sentiment_analysis(self):
        """Perform VADER sentiment analysis on the dataset."""
        if 'text_column' in self.data.columns and self.data['text_column'].str.len().mean() > 5:  # Checking for suitable length
            print("Performing VADER sentiment analysis...")
            for index, row in self.data.iterrows():
                text = row['text_column']
                sentiment_scores = self.analyzer.polarity_scores(text)
                print(f"Text: {text}\nSentiment Scores: {sentiment_scores}")
        else:
            print("Looking for text data in your dataset…")
            print("Sorry, your dataset does not have a suitable length text data.")
            print("Therefore, Sentiment Analysis is not possible. Returning to previous menu...")

    def textblob_sentiment_analysis(self):
        """Perform TextBlob sentiment analysis on the dataset."""
        if 'text_column' in self.data.columns and self.data['text_column'].str.len().mean() > 5:  # Checking for suitable length
            print("Performing TextBlob sentiment analysis...")
            for index, row in self.data.iterrows():
                text = row['text_column']
                blob = TextBlob(text)
                sentiment = blob.sentiment
                print(f"Text: {text}\nPolarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
        else:
            print("Looking for text data in your dataset…")
            print("Sorry, your dataset does not have a suitable length text data.")
            print("Therefore, Sentiment Analysis is not possible. Returning to previous menu...")
