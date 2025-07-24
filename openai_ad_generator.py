"""
OpenAI-Powered Ad Generator with RAG Integration
Generates context-aware ad creative using real OpenAI API
"""
import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from utils.rag_processor import RAGProcessor
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Or set environment variables manually")

class OpenAIAdGenerator:
    # Model Configuration - Using the latest and best OpenAI models
    TEXT_MODEL = "gpt-4o"  # Latest GPT-4 Omni model (May 2024) - best for reasoning and complex tasks
    IMAGE_MODEL = "dall-e-3"  # Latest DALL-E 3 model - best for image generation
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI Ad Generator
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment variable OPENAI_API_KEY)
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-openai-api-key-here':
            print("⚠️  No OpenAI API key found. Please:")
            print("   1. Edit .env file and replace 'your-openai-api-key-here' with your actual key")
            print("   2. Or set environment variable: export OPENAI_API_KEY='sk-your-key-here'")
            print("   3. Or pass api_key parameter directly")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized successfully")
        
        self.rag_processor = None
    
    def load_rag_processor(self):
        """Load RAG processor for enhanced context"""
        self.rag_processor = RAGProcessor()
        if not self.rag_processor.load_index():
            print("Warning: No vector database found. Some features may be limited.")
    
    def create_system_prompt(self) -> str:
        """Create system prompt for ad generation"""
        return """You are a creative marketing assistant for leading global asset management firms. Your task is to generate compelling, compliant ad creative that connects client investment insights with current market news.

Key Requirements:
- Maintain professional, authoritative, and accessible tone
- Ensure compliance with financial industry regulations
- Focus on thought leadership and valuable insights
- Avoid specific performance guarantees or overly promotional language
- Create timely, relevant content that resonates with institutional investors

Ad Format Guidelines:
- LinkedIn Single Image: Headline (150 chars max), Body (600 chars max), CTA, Image description
- Banner Ad 300x250: Headline (50 chars max), Body (100 chars max), CTA, Image description
- All content should demonstrate clear connection between client message and news themes

Output format: Return valid JSON with structured ad content."""
    
    def create_ad_prompt(self, client_data: Dict[str, Any], relevant_news: List[Dict[str, Any]]) -> str:
        """
        Create detailed prompt for ad generation
        
        Args:
            client_data: Client information and landing page content
            relevant_news: List of relevant news articles with similarity scores
            
        Returns:
            Formatted prompt for OpenAI
        """
        # Extract key information
        client_name = client_data.get('client_name', 'Asset Management Firm')
        landing_page_content = client_data.get('landing_page_content', '')
        keywords = client_data.get('landing_page_keywords', [])
        
        # Format landing page summary
        landing_summary = landing_page_content[:800] if landing_page_content else "Investment expertise and market insights"
        
        # Format top news articles (use all relevant news found)
        news_context = []
        for i, news in enumerate(relevant_news):
            news_context.append(f"{i+1}. {news.get('title', 'Market Development')} (Source: {news.get('source', 'Financial News')}) - Relevance: {news.get('similarity_score', 0.5):.2f}")
        
        # Enhanced keyword context
        keyword_context = f"**AI-Enhanced Keywords:** {', '.join(keywords[:8])}" if keywords else "**Keywords:** Not available"
        
        prompt = f"""
**Client Context:**
Client: {client_name}
Landing Page URL: {client_data.get('url', '')}

**Key Themes & Expertise:**
{landing_summary}

{keyword_context}

**Relevant Market News (Ranked by AI Semantic Matching):**
{chr(10).join(news_context)}

**Task:** Generate ad creative using the AI-enhanced keywords and semantic news matching that meaningfully connects the client's expertise with the current news landscape:

1. **LinkedIn Single Image Ad:**
   - Headline (max 150 characters)
   - Body (max 600 characters) 
   - Call-to-Action
   - Image Description (detailed visual concept)

2. **Banner Ad 300x250:**
   - Headline (max 50 characters)
   - Body (max 100 characters)
   - Call-to-Action
   - Image Description (detailed visual concept)

3. **Additional Creative Concept:**
   - Provide one additional innovative ad format or approach

**Requirements:**
- Connect client expertise with at least one news item
- Maintain compliance and professional tone
- Focus on thought leadership, not direct selling
- Ensure headlines are compelling and news-responsive
- Make the connection between news and client value clear

**Output:** Return as JSON with keys: "linkedin_single_image", "banner_ad_300x250", "additional_creative", each containing "headline", "body", "call_to_action", "image_description", and "news_connection_rationale".
"""
        return prompt
    
    def generate_ad_creative(self, client_data: Dict[str, Any], relevant_news: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate ad creative using OpenAI API
        
        Args:
            client_data: Client information
            relevant_news: Relevant news articles
            
        Returns:
            Generated ad creative or None if failed
        """
        if not self.client:
            return self._generate_mock_response(client_data, relevant_news)
        
        try:
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_ad_prompt(client_data, relevant_news)
            
            response = self.client.chat.completions.create(
                model=self.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    return json.loads(json_content)
                else:
                    # If no JSON found, create structured response
                    return self._parse_text_response(content, client_data, relevant_news)
            except json.JSONDecodeError:
                return self._parse_text_response(content, client_data, relevant_news)
                
        except Exception as e:
            print(f"Error generating ad creative: {e}")
            return self._generate_mock_response(client_data, relevant_news)
    
    def _generate_mock_response(self, client_data: Dict[str, Any], relevant_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mock response when API is not available"""
        client_name = client_data.get('client_name', 'Client')
        keywords = client_data.get('landing_page_keywords', ['investment', 'finance'])
        top_news = relevant_news[0] if relevant_news else {'title': 'Market Developments', 'source': 'Financial News'}
        
        return {
            "linkedin_single_image": {
                "headline": f"[MOCK] How {client_name}'s {keywords[0] if keywords else 'expertise'} addresses today's market challenges",
                "body": f"[MOCK] In light of recent developments including '{top_news['title'][:50]}...', discover our insights on navigating market volatility. Our proven approach to {keywords[0] if keywords else 'investment management'} helps institutional clients stay ahead of market trends.",
                "call_to_action": "Learn More",
                "image_description": "[MOCK] Professional image of diverse team analyzing market data on multiple screens, with subtle overlay of upward trending financial charts in blue and green tones.",
                "news_connection_rationale": f"[MOCK] Connects client expertise in {keywords[0] if keywords else 'finance'} with current market news about {top_news['title'][:30]}..."
            },
            "banner_ad_300x250": {
                "headline": f"[MOCK] Navigate {top_news['title'].split()[0]} Trends",
                "body": f"[MOCK] Expert insights on {keywords[0] if keywords else 'market'} strategy.",
                "call_to_action": "Explore",
                "image_description": "[MOCK] Clean, modern graphic with ascending arrow and financial symbols.",
                "news_connection_rationale": f"[MOCK] Brief format connects news about {top_news['title'][:20]}... with client capabilities"
            },
            "additional_creative": {
                "headline": f"[MOCK] {client_name} Perspective: Market Outlook 2025",
                "body": f"[MOCK] Our research team analyzes the implications of recent {top_news['title'][:30]}... for institutional portfolios.",
                "call_to_action": "Download Report",
                "image_description": "[MOCK] Sophisticated infographic showing market trends with client branding.",
                "news_connection_rationale": f"[MOCK] Positions client as thought leader responding to {top_news['title']}"
            }
        }
    
    def _parse_text_response(self, content: str, client_data: Dict[str, Any], relevant_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse text response if JSON parsing fails"""
        # This is a fallback - in practice, you'd implement more sophisticated parsing
        return self._generate_mock_response(client_data, relevant_news)
    
    def enhance_with_rag(self, client_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use RAG to find additional relevant context
        
        Args:
            client_data: Client information
            
        Returns:
            Enhanced list of relevant news and context
        """
        if not self.rag_processor:
            return client_data.get('relevant_news', [])
        
        # Use RAG to find additional relevant content
        client_name = client_data.get('client_name', '')
        landing_content = client_data.get('landing_page_content', '')
        
        # Extract themes for enhanced search
        keywords = client_data.get('landing_page_keywords', [])
        search_query = f"{' '.join(keywords[:3])} {landing_content[:200]}"
        
        # Get additional context
        enhanced_results = self.rag_processor.semantic_search(search_query, k=8, filter_type='news_article')
        
        return enhanced_results
    
    def generate_campaign_for_client(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete ad campaign for a client
        
        Args:
            client_data: Client information with relevant news
            
        Returns:
            Complete campaign with multiple ad formats
        """
        print(f"\n🎯 Generating ads for {client_data.get('client_name', 'Client')}")
        
        # Enhance with RAG if available
        if self.rag_processor:
            relevant_news = self.enhance_with_rag(client_data)
        else:
            relevant_news = client_data.get('relevant_news', [])
        
        # Generate primary ad creative (use all relevant news found by RAG)
        primary_ads = self.generate_ad_creative(client_data, relevant_news)
        
        # Add metadata
        campaign = {
            'client_name': client_data.get('client_name'),
            'client_url': client_data.get('url'),
            'generation_timestamp': time.time(),
            'relevant_news_count': len(relevant_news),
            'top_keywords': client_data.get('landing_page_keywords', [])[:5],
            'ad_creative': primary_ads,
            'news_sources': [
                {
                    'title': news.get('title', ''),
                    'source': news.get('source', ''),
                    'relevance_score': news.get('similarity_score', 0)
                }
                for news in relevant_news[:5]
            ]
        }
        
        print(f"  ✓ Generated {len(primary_ads) if primary_ads else 0} ad formats")
        print(f"  ✓ Used {len(relevant_news)} relevant news articles")
        
        return campaign

def generate_complete_campaign(processed_data_file: str = 'processed_client_data_rag.json') -> List[Dict[str, Any]]:
    """
    Generate complete ad campaigns for all clients
    
    Args:
        processed_data_file: Path to processed client data
        
    Returns:
        List of complete campaigns
    """
    # Load processed data
    try:
        with open(processed_data_file, 'r', encoding='utf-8') as f:
            processed_clients = json.load(f)
    except FileNotFoundError:
        print("Please run rag_processor.py first to generate processed data")
        return []
    
    # Initialize generator
    generator = OpenAIAdGenerator()  # Using placeholder API key
    generator.load_rag_processor()
    
    # Generate campaigns
    campaigns = []
    print("=== GENERATING AD CAMPAIGNS ===")
    
    for client_data in processed_clients:
        try:
            campaign = generator.generate_campaign_for_client(client_data)
            campaigns.append(campaign)
            
            # Add small delay to be respectful if using real API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error generating campaign for {client_data.get('client_name', 'Unknown')}: {e}")
    
    return campaigns

if __name__ == "__main__":
    # Generate complete campaigns
    campaigns = generate_complete_campaign()
    
    # Save campaigns
    with open('generated_ad_campaigns.json', 'w', encoding='utf-8') as f:
        json.dump(campaigns, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(campaigns)} ad campaigns")
    print("📁 Campaigns saved to generated_ad_campaigns.json")
    
    # Print summary
    print(f"\n=== CAMPAIGN SUMMARY ===")
    for campaign in campaigns:
        client_name = campaign.get('client_name', 'Unknown')
        ad_count = len(campaign.get('ad_creative', {}))
        news_count = campaign.get('relevant_news_count', 0)
        
        print(f"\n{client_name}:")
        print(f"  📢 Ad formats: {ad_count}")
        print(f"  📰 News articles analyzed: {news_count}")
        print(f"  🔑 Top keywords: {', '.join(campaign.get('top_keywords', [])[:3])}")
        
        # Show sample ad
        ad_creative = campaign.get('ad_creative', {})
        if ad_creative and 'linkedin_single_image' in ad_creative:
            linkedin_ad = ad_creative['linkedin_single_image']
            print(f"  💡 Sample headline: '{linkedin_ad.get('headline', '')[:60]}...'")
        
        # Show top news source
        news_sources = campaign.get('news_sources', [])
        if news_sources:
            top_news = news_sources[0]
            print(f"  📊 Top relevant news: '{top_news.get('title', '')[:50]}...' (score: {top_news.get('relevance_score', 0):.2f})")