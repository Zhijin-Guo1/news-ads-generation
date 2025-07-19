from sentence_transformers import SentenceTransformer
import json
from rake_nltk import Rake

class NLPProcessor:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rake = Rake()

    def get_embedding(self, text):
        return self.model.encode(text).tolist()

    def extract_keywords(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()[:5] # Get top 5 ranked phrases as keywords

if __name__ == "__main__":
    processor = NLPProcessor()

    # Load parsed client data
    with open("/home/ubuntu/parsed_client_data.json", "r") as f:
        client_data = json.load(f)

    # Process each client entry
    processed_client_data = []
    for entry in client_data:
        url = entry["url"]
        news_articles = entry["news_articles"]

        # For demonstration, we'll use the scraped content from the test URL
        # In a real scenario, you would scrape each client URL here
        if url == "https://www.ssga.com/uk/en_gb/institutional/capabilities/esg":
            with open("/home/ubuntu/test_scraped_content.txt", "r", encoding="utf-8") as f:
                landing_page_text = f.read()
        else:
            landing_page_text = "Placeholder text for " + url # Replace with actual scraping

        landing_page_embedding = processor.get_embedding(landing_page_text)
        landing_page_keywords = processor.extract_keywords(landing_page_text)

        processed_news_articles = []
        for article in news_articles:
            article_text = article["title"] + " " + article["source"] # Combine title and source for embedding
            article_embedding = processor.get_embedding(article_text)
            article_keywords = processor.extract_keywords(article_text)
            processed_news_articles.append({
                **article,
                "embedding": article_embedding,
                "keywords": article_keywords
            })

        processed_client_data.append({
            "url": url,
            "landing_page_text": landing_page_text,
            "landing_page_embedding": landing_page_embedding,
            "landing_page_keywords": landing_page_keywords,
            "news_articles": processed_news_articles
        })

    with open("/home/ubuntu/processed_client_data.json", "w") as f:
        json.dump(processed_client_data, f, indent=4)
    print("Processed client data saved to /home/ubuntu/processed_client_data.json")


