import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_ads(processed_data):
    ad_outputs = []

    for entry in processed_data:
        client_url = entry["url"]
        landing_page_text = entry["landing_page_text"]
        landing_page_embedding = np.array(entry["landing_page_embedding"]).reshape(1, -1)
        landing_page_keywords = entry["landing_page_keywords"]
        news_articles = entry["news_articles"]

        # Calculate similarity and rank news articles
        ranked_news = []
        for article in news_articles:
            article_embedding = np.array(article["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(landing_page_embedding, article_embedding)[0][0]
            ranked_news.append({
                **article,
                "similarity": similarity
            })
        
        # Sort by similarity in descending order
        ranked_news = sorted(ranked_news, key=lambda x: x["similarity"], reverse=True)

        # Select top N relevant news articles (e.g., top 3)
        top_n_news = ranked_news[:3]

        # Simulate LLM interaction for ad generation
        # In a real scenario, this would be an API call to an LLM
        # For now, we'll construct a plausible output based on the problem description
        
        # Helper function to safely get keyword or default
        def get_keyword(keywords, index, default_text):
            return keywords[index] if len(keywords) > index else default_text

        # Helper function to safely get news title or default
        def get_news_title(news_list, index, default_text):
            return news_list[index]["title"] if len(news_list) > index else default_text

        simulated_ad_creative = {
            "client_url": client_url,
            "landing_page_summary": landing_page_text[:200] + "..." if len(landing_page_text) > 200 else landing_page_text,
            "relevant_news": [{
                "title": news["title"],
                "source": news["source"],
                "similarity": news["similarity"]
            } for news in top_n_news],
            "ad_formats": {
                "linkedin_single_image": {
                    "headline": f"[LLM Generated] Discover how {get_keyword(landing_page_keywords, 0, 'our solutions')} align with current market trends.",
                    "body": f"[LLM Generated] In light of recent news regarding {get_news_title(top_n_news, 0, 'market developments')}, explore our insights on {get_keyword(landing_page_keywords, 0, 'investment strategies')}. Learn more about our approach to {get_keyword(landing_page_keywords, 1, 'sustainable investing')} and how it can benefit your portfolio.",
                    "call_to_action": "Learn More",
                    "imagery_suggestion": "[LLM Generated] Image of a diverse group of professionals collaborating in a modern office, with financial charts subtly overlaid in the background. Focus on innovation and growth. Aspect Ratio: Square."
                },
                "banner_ad_300x250": {
                    "headline": f"[LLM Generated] Market Insights: {get_news_title(top_n_news, 0, 'Stay Ahead')}",
                    "body": f"[LLM Generated] Align your strategy with current {get_keyword(landing_page_keywords, 0, 'market dynamics')}.",
                    "call_to_action": "Explore Now",
                    "imagery_suggestion": "[LLM Generated] Abstract financial graphic with upward trending lines, subtle blue and green color palette. Aspect Ratio: Square."
                }
            }
        }
        ad_outputs.append(simulated_ad_creative)

    return ad_outputs

if __name__ == "__main__":
    with open("/home/ubuntu/processed_client_data.json", "r") as f:
        processed_data = json.load(f)

    generated_ads = generate_ads(processed_data)

    with open("/home/ubuntu/generated_ads.json", "w") as f:
        json.dump(generated_ads, f, indent=4)
    print("Generated ads saved to /home/ubuntu/generated_ads.json")


