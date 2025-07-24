"""
RAG-Enabled NLP Processor for Alphix ML Challenge
Implements vector database and semantic search for news-responsive ad generation
Enhanced with OpenAI-powered keyword extraction
"""
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import nltk
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import time

# Set NLTK data path to local directory first
local_nltk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
if local_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk_path)

# Download NLTK stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Local NLTK data not found, downloading...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

from rake_nltk import Rake

class RAGProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", openai_api_key: str = None):
        """
        Initialize RAG processor with embedding model and vector database
        
        Args:
            model_name: SentenceTransformer model name
            openai_api_key: OpenAI API key for enhanced keyword extraction
        """
        self.model = SentenceTransformer(model_name)
        self.rake = Rake()  # Keep as fallback
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.news_database = []
        self.client_database = []
        
        # Initialize OpenAI for enhanced keyword extraction
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                print(f"Warning: OpenAI client not initialized: {e}")
                print("Falling back to RAKE for keyword extraction")
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.model.encode(text)
    
    def extract_keywords(self, text: str = None, max_keywords: int = 5, url: str = None) -> List[str]:
        """Extract keywords using real-time web scraping and GPT-4o function calling for maximum accuracy"""
        if self.openai_client and url:
            return self._extract_keywords_realtime_web(url, max_keywords)
        elif self.openai_client and text:
            return self._extract_keywords_openai(text, max_keywords)
        else:
            return self._extract_keywords_rake(text or "", max_keywords)
    
    def _extract_keywords_realtime_web(self, url: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords using real-time web scraping and GPT-4o function calling"""
        try:
            print(f"🌐 Real-time scraping: {url}")
            
            # Step 1: Real-time web scraping
            scraped_content = self._scrape_webpage_realtime(url)
            if not scraped_content or scraped_content.startswith("Error"):
                print(f"❌ Web scraping failed: {scraped_content}")
                return []
            
            print(f"✅ Scraped {len(scraped_content)} characters from live webpage")
            
            # Step 2: GPT-4o function calling with fresh content
            return self._extract_keywords_with_function_calling(scraped_content, url, max_keywords)
            
        except Exception as e:
            print(f"❌ Real-time keyword extraction failed: {e}")
            return []
    
    def _scrape_webpage_realtime(self, url: str) -> str:
        """Perform real-time web scraping during AI processing"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            print(f"   📡 Fetching live content from {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length for AI processing
            if len(text) > 8000:
                text = text[:8000]
            
            return text
            
        except requests.exceptions.Timeout:
            return f"Error: Timeout while accessing {url}"
        except requests.exceptions.RequestException as e:
            return f"Error: Network error - {str(e)}"
        except Exception as e:
            return f"Error: Scraping failed - {str(e)}"
    
    def _extract_keywords_with_function_calling(self, content: str, url: str, max_keywords: int = 5) -> List[str]:
        """Use GPT-4o function calling to analyze fresh web content for investment themes"""
        try:
            # Define the function for keyword extraction
            keyword_function = {
                "name": "extract_investment_keywords",
                "description": "Extract strategic investment keywords from financial services webpage content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of exactly {max_keywords} strategic investment keywords/phrases (1-4 words each) that capture the firm's core investment focus and expertise"
                        },
                        "analysis_summary": {
                            "type": "string",
                            "description": "Brief summary of the investment themes and market focus identified"
                        }
                    },
                    "required": ["keywords", "analysis_summary"]
                }
            }
            
            prompt = f"""
            You are analyzing LIVE, CURRENT content from a financial services webpage to extract the most strategically important investment keywords.
            
            URL: {url}
            Fresh webpage content (just scraped): {content[:4000]}
            
            As a world-class financial services expert, identify the {max_keywords} most important keywords that represent:
            
            1. CORE INVESTMENT PHILOSOPHY: What fundamental approach defines this firm?
            2. STRATEGIC MARKET FOCUS: Which markets, sectors, or themes do they specialize in?
            3. CURRENT POSITIONING: How are they positioned for today's market conditions?
            4. COMPETITIVE ADVANTAGES: What makes them unique in their space?
            5. EXPERTISE DOMAINS: What are their recognized areas of deep knowledge?
            
            Focus on investment-relevant terms that would help match this firm with relevant financial news.
            Each keyword should be 1-4 words and capture strategic investment themes.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a world-class financial services marketing strategist and investment analyst with deep expertise in identifying strategic investment themes from current market communications."},
                    {"role": "user", "content": prompt}
                ],
                functions=[keyword_function],
                function_call={"name": "extract_investment_keywords"},
                temperature=0.1
            )
            
            # Parse function call result
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "extract_investment_keywords":
                import json
                result = json.loads(function_call.arguments)
                keywords = result.get("keywords", [])
                analysis = result.get("analysis_summary", "")
                
                print(f"   🧠 AI Analysis: {analysis}")
                print(f"   🎯 Extracted keywords: {keywords}")
                
                return keywords[:max_keywords]
            else:
                print("❌ Function calling failed, falling back to direct extraction")
                return self._extract_keywords_openai(content, max_keywords)
                
        except Exception as e:
            print(f"❌ Function calling extraction failed: {e}")
            # Fallback to standard OpenAI extraction
            return self._extract_keywords_openai(content, max_keywords)
    
    
    
    def _extract_keywords_openai(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords using most advanced OpenAI model for maximum financial domain accuracy"""
        try:
            prompt = f"""
            You are the world's leading expert in financial services marketing and investment analysis. Analyze the following financial services text with deep expertise to extract the most strategically important keywords that represent the company's core investment philosophy, unique competitive advantages, and market positioning.
            
            Your analysis should identify:
            1. CORE INVESTMENT PHILOSOPHY: What fundamental investment approach defines this firm?
            2. STRATEGIC DIFFERENTIATORS: What makes this firm unique in the market?
            3. TARGET MARKET FOCUS: Which specific market segments, asset classes, or investment themes?
            4. CURRENT POSITIONING: How is the firm positioned relative to current market conditions?
            5. EXPERTISE DOMAINS: What are their recognized areas of deep expertise?
            
            Text to analyze:
            {text[:4000]}  # Increased limit for more context
            
            Extract exactly {max_keywords} highly strategic keywords/phrases that capture the essence of this firm's investment identity. Each should be 1-4 words and directly relevant to investment marketing.
            
            Return ONLY the keywords, one per line, no numbering or formatting:
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Most advanced model for maximum accuracy
                messages=[
                    {"role": "system", "content": "You are a world-class financial services marketing strategist and investment analyst with 20+ years of experience in asset management."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1  # Lower temperature for more focused, accurate results
            )
            
            keywords = response.choices[0].message.content.strip().split('\n')
            # Clean and filter keywords with higher quality standards
            clean_keywords = []
            for keyword in keywords:
                keyword = keyword.strip().strip('-').strip('•').strip('*').strip()
                # More rigorous filtering for investment relevance
                if (keyword and len(keyword) > 2 and len(keyword) < 50 and 
                    not keyword.lower().startswith(('the ', 'and ', 'or ', 'but ', 'with '))):
                    clean_keywords.append(keyword)
            
            return clean_keywords[:max_keywords]
            
        except Exception as e:
            print(f"Advanced OpenAI keyword extraction failed: {e}")
            return self._extract_keywords_rake(text, max_keywords)
    
    def _extract_keywords_rake(self, text: str, max_keywords: int = 5) -> List[str]:
        """Fallback keyword extraction using RAKE"""
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()[:max_keywords]
    
    def build_vector_database(self, client_data: List[Dict[str, Any]]) -> None:
        """
        Build FAISS vector database from client and news data
        Each client's news articles are kept separate for client-specific search
        
        Args:
            client_data: List of client data with landing page content and news articles
        """
        print("Building vector database...")
        
        # Prepare embeddings for all content
        embeddings = []
        metadata = []
        
        # Process client landing pages
        for client in client_data:
            if client.get('landing_page_content'):
                # Normalize client name
                normalized_name = self._normalize_client_name(client['client_name'])
                
                # Chunk landing page content for better retrieval
                chunks = self._chunk_text(client['landing_page_content'], max_length=512)
                
                for i, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    embeddings.append(embedding)
                    metadata.append({
                        'type': 'landing_page',
                        'client_name': normalized_name,
                        'url': client['url'],
                        'chunk_id': i,
                        'content': chunk,
                        'keywords': self.extract_keywords(chunk, url=client['url'])
                    })
        
        # Process news articles - IMPORTANT: Keep client association for separate searching
        for client in client_data:
            # Normalize client name
            normalized_name = self._normalize_client_name(client['client_name'])
            
            for article in client['news_articles']:
                # Combine title and source for richer embedding
                article_text = f"{article['title']} {article.get('source', '')}"
                embedding = self.get_embedding(article_text)
                embeddings.append(embedding)
                metadata.append({
                    'type': 'news_article',
                    'client_name': normalized_name,  # Key: This maintains client association with normalized name
                    'title': article['title'],
                    'source': article.get('source', ''),
                    'published_date': article.get('published_date', ''),
                    'url': article.get('url', ''),
                    'content': article_text,
                    'keywords': self.extract_keywords(article_text)
                })
        
        # Build FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata = metadata
        
        print(f"Vector database built with {len(embeddings)} embeddings")
        
        # Save index and metadata
        self._save_index()
    
    def _chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Chunk text into smaller pieces for better retrieval"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _normalize_client_name(self, client_name: str) -> str:
        """
        Normalize client names to handle variations (e.g., 'T. Rowe Price' vs 'T Rowe Price')
        """
        if not client_name:
            return client_name
        
        # Handle T. Rowe Price variations
        if 'T. Rowe Price' in client_name or 'T.Rowe Price' in client_name:
            return 'T Rowe Price'
        
        return client_name.strip()

    def semantic_search(self, query: str, k: int = 5, filter_type: str = None, filter_client: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search in vector database
        
        Args:
            query: Search query
            k: Number of results to return
            filter_type: Optional filter by content type ('landing_page' or 'news_article')
            filter_client: Optional filter by client name for client-specific search
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if self.index is None:
            raise ValueError("Vector database not built yet. Call build_vector_database() first.")
        
        # Normalize filter_client name if provided
        if filter_client:
            filter_client = self._normalize_client_name(filter_client)
        
        # Get query embedding
        query_embedding = self.get_embedding(query).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search with more results to allow for filtering and ensure we get k results
        search_k = k * 5 if filter_client else k * 2  # Increased multiplier
        scores, indices = self.index.search(query_embedding, min(search_k, len(self.metadata)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            metadata = self.metadata[idx]
            
            # Apply type filter if specified
            if filter_type and metadata['type'] != filter_type:
                continue
            
            # Apply client filter if specified (with name normalization)
            if filter_client:
                found_name = self._normalize_client_name(metadata['client_name'])
                if found_name != filter_client:
                    continue
            
            results.append({
                **metadata,
                'similarity_score': float(score)
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def find_relevant_news(self, client_name: str, landing_page_content: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Find relevant news articles using enhanced OpenAI-powered keyword extraction
        
        Args:
            client_name: Name of the client
            landing_page_content: Content of the landing page
            k: Number of relevant news articles to return
            
        Returns:
            List of exactly k most relevant news articles with improved accuracy
        """
        # Normalize client name to handle variations
        normalized_client_name = self._normalize_client_name(client_name)
        
        # Extract investment themes and market views from landing page
        investment_themes = self._extract_investment_themes(landing_page_content)
        
        # Extract keywords using real-time web scraping
        client_url = None
        # Try to get URL from the processed data structure
        if hasattr(self, 'metadata'):
            for item in self.metadata:
                if (item.get('type') == 'landing_page' and 
                    item.get('client_name') == normalized_client_name):
                    client_url = item.get('url')
                    break
        
        print(f"🌐 Using real-time web scraping for {client_name} at {client_url}")
        landing_page_keywords = self.extract_keywords(url=client_url, max_keywords=8)
        
        # Create a focused query based on investment themes and AI-extracted keywords
        query = self._create_thematic_query(investment_themes, normalized_client_name, landing_page_keywords)
        
        print(f"Enhanced query for {client_name}: {query}")
        print(f"OpenAI keywords: {landing_page_keywords}")
        
        # Search within this client's news articles with increased search space to ensure k results
        results = self.semantic_search(
            query, 
            k=k * 2,  # Search for more to ensure we get k results after filtering
            filter_type='news_article',
            filter_client=normalized_client_name
        )
        
        # If we still don't have enough results, try a broader search
        if len(results) < k:
            # Fallback: search with just client name and basic terms
            fallback_query = f"{normalized_client_name} investment finance market economic"
            fallback_results = self.semantic_search(
                fallback_query,
                k=k * 3,  # Even broader search
                filter_type='news_article',
                filter_client=normalized_client_name
            )
            
            # Combine results, removing duplicates by title
            seen_titles = set(r['title'] for r in results)
            for result in fallback_results:
                if result['title'] not in seen_titles and len(results) < k:
                    results.append(result)
                    seen_titles.add(result['title'])
        
        # Return exactly k results (or all available if less than k)
        return results[:k]
    
    def _extract_investment_themes(self, landing_page_content: str) -> Dict[str, str]:
        """
        Extract investment themes and market views from landing page content
        
        Returns:
            Dictionary with investment themes, market outlook, key focus areas
        """
        content_lower = landing_page_content.lower()
        
        # Define theme extraction patterns
        theme_patterns = {
            'monetary_policy': ['federal reserve', 'fed policy', 'interest rates', 'rate cuts', 'inflation', 'monetary policy'],
            'market_outlook': ['market outlook', 'economic outlook', '2025 outlook', 'market trends', 'investment outlook'],
            'investment_strategy': ['investment strategy', 'asset allocation', 'portfolio', 'diversification', 'risk management'],
            'sustainability': ['sustainable investing', 'esg', 'environmental', 'social', 'governance', 'climate'],
            'sectors': ['emerging markets', 'equities', 'bonds', 'real estate', 'technology', 'healthcare'],
            'economic_themes': ['inflation', 'growth', 'recession', 'recovery', 'volatility', 'uncertainty']
        }
        
        # Extract themes present in the content
        detected_themes = {}
        for theme_name, keywords in theme_patterns.items():
            theme_content = []
            for keyword in keywords:
                if keyword in content_lower:
                    # Find sentences containing this keyword
                    sentences = landing_page_content.split('.')
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower() and len(sentence.strip()) > 20:
                            theme_content.append(sentence.strip())
                            break  # One sentence per keyword is enough
            
            if theme_content:
                detected_themes[theme_name] = ' '.join(theme_content[:2])  # Max 2 sentences per theme
        
        return detected_themes
    
    def _create_thematic_query(self, investment_themes: Dict[str, str], client_name: str = None, landing_page_keywords: List[str] = None) -> str:
        """
        Create a focused query based on real-time web scraping keywords
        
        Args:
            investment_themes: Dictionary of detected themes and their content
            client_name: Name of the client for context
            landing_page_keywords: Real-time web scraping extracted keywords
            
        Returns:
            Optimized query string for news matching within client's collection
        """
        if self.openai_client and landing_page_keywords:
            return self._create_realtime_enhanced_query(client_name, landing_page_keywords)
        else:
            return self._create_traditional_query(investment_themes, client_name)
    
    def _create_realtime_enhanced_query(self, client_name: str, landing_page_keywords: List[str]) -> str:
        """Create query using real-time web scraping keywords for maximum accuracy"""
        try:
            prompt = f"""
            You are the world's leading expert in financial markets and investment strategy. 
            Create the most strategically accurate search query to find financial news articles 
            that would be highly relevant to {client_name}'s current investment focus and expertise.
            
            REAL-TIME EXTRACTED KEYWORDS (from live webpage): {', '.join(landing_page_keywords)}
            
            Based on these fresh, current keywords that represent the firm's actual positioning today, 
            create a precise search query (maximum 8 strategically chosen words) that would return 
            the most relevant financial news.
            
            Focus on the core investment themes and market positioning revealed by the real-time analysis.
            
            Return ONLY the optimized search query:
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a world-class financial markets strategist specializing in semantic search optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            ai_query = response.choices[0].message.content.strip()
            
            # Validate and clean the query
            query_words = ai_query.split()
            if len(query_words) > 8:
                query_words = query_words[:8]
            
            enhanced_query = ' '.join(query_words)
            print(f"  🧠 Real-time enhanced query: '{enhanced_query}'")
            
            return enhanced_query
            
        except Exception as e:
            print(f"Real-time query generation failed: {e}")
            # Simple fallback using keywords directly
            return ' '.join(landing_page_keywords[:6]) if landing_page_keywords else "investment finance market"
    
    def _create_traditional_query(self, investment_themes: Dict[str, str], client_name: str) -> str:
        """Traditional query creation as fallback"""
        # Always use client-specific fallback for more reliable results
        if client_name:
            if 'PIMCO' in client_name:
                base_query = "federal reserve interest rates monetary policy inflation economic"
            elif 'State Street' in client_name:
                base_query = "sustainable investing ESG environmental social governance climate"
            elif 'T Rowe Price' in client_name or 'T. Rowe Price' in client_name:
                base_query = "emerging markets investment strategy portfolio management equity"
            else:
                base_query = "investment finance market economic outlook"
        else:
            base_query = "investment finance market economic outlook"
        
        # If we have themes, enhance the base query with relevant keywords
        if investment_themes:
            # Define high-quality financial keywords for each theme
            theme_keywords = {
                'monetary_policy': ['federal reserve', 'interest rates', 'inflation', 'monetary policy', 'rate cuts'],
                'market_outlook': ['market outlook', 'economic outlook', 'market trends', 'investment outlook'],
                'investment_strategy': ['investment strategy', 'portfolio', 'asset allocation', 'risk management'],
                'sustainability': ['ESG', 'sustainable investing', 'climate', 'environmental', 'governance'],
                'sectors': ['emerging markets', 'equities', 'bonds', 'technology', 'healthcare'],
                'economic_themes': ['inflation', 'growth', 'recession', 'volatility', 'economic']
            }
            
            # Collect relevant keywords from detected themes
            enhancement_keywords = []
            for theme_name in investment_themes.keys():
                if theme_name in theme_keywords:
                    enhancement_keywords.extend(theme_keywords[theme_name][:2])  # Top 2 keywords per theme
            
            # Add enhancement keywords to base query
            if enhancement_keywords:
                # Remove duplicates while preserving order
                all_keywords = base_query.split() + enhancement_keywords
                unique_keywords = []
                seen = set()
                for keyword in all_keywords:
                    if keyword.lower() not in seen:
                        unique_keywords.append(keyword)
                        seen.add(keyword.lower())
                
                # Limit to 8 most relevant keywords
                final_query = ' '.join(unique_keywords[:8])
            else:
                final_query = base_query
        else:
            final_query = base_query
            
        return final_query
    
    def get_contextual_information(self, client_name: str, topic: str, k: int = 5) -> Dict[str, Any]:
        """
        Get contextual information for ad generation
        Now searches within client-specific content only
        
        Args:
            client_name: Name of the client
            topic: Topic or theme to focus on
            k: Number of results per category
            
        Returns:
            Dictionary with landing page context and relevant news (client-specific)
        """
        # Get landing page context for this specific client
        landing_page_results = self.semantic_search(
            f"{client_name} {topic}", 
            k=k, 
            filter_type='landing_page',
            filter_client=client_name
        )
        
        # Get relevant news from this client's news only
        news_results = self.semantic_search(
            f"{topic} finance investment", 
            k=k, 
            filter_type='news_article',
            filter_client=client_name  # KEY CHANGE: Only this client's news
        )
        
        return {
            'landing_page_context': landing_page_results,
            'relevant_news': news_results,
            'topic': topic,
            'client_name': client_name
        }
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, "vector_index.faiss")
        with open("vector_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print("Vector database saved to disk")
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if os.path.exists("vector_index.faiss") and os.path.exists("vector_metadata.pkl"):
            self.index = faiss.read_index("vector_index.faiss")
            with open("vector_metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            print("Vector database loaded from disk")
            return True
        return False

def process_client_data_with_rag(client_data_file: str = 'client_data_with_content.json', openai_api_key: str = None) -> Dict[str, Any]:
    """
    Process client data and build RAG system with enhanced OpenAI keyword extraction
    
    Args:
        client_data_file: Path to client data file
        openai_api_key: OpenAI API key for enhanced keyword extraction
        
    Returns:
        Processed data with RAG capabilities
    """
    # Load client data
    with open(client_data_file, 'r', encoding='utf-8') as f:
        client_data = json.load(f)
    
    # Initialize RAG processor with OpenAI support
    rag_processor = RAGProcessor(openai_api_key=openai_api_key)
    
    # Try to load existing index
    if not rag_processor.load_index():
        # Build new vector database
        rag_processor.build_vector_database(client_data)
    
    # Process each client
    processed_data = []
    for client in client_data:
        client_name = client['client_name']
        landing_page_content = client.get('landing_page_content', '')
        
        if not landing_page_content:
            print(f"Warning: No landing page content for {client_name}")
            continue
        
        # Extract keywords using real-time web scraping and GPT-4o function calling
        print(f"🌐 Real-time keyword extraction for {client_name}")
        keywords = rag_processor.extract_keywords(url=client.get('url'), max_keywords=8)
        
        # Find relevant news articles
        relevant_news = rag_processor.find_relevant_news(client_name, landing_page_content)
        
        # Create processed entry
        processed_entry = {
            'client_name': client_name,
            'url': client['url'],
            'landing_page_content': landing_page_content,
            'landing_page_keywords': keywords,
            'relevant_news': relevant_news,
            'rag_processor': rag_processor  # Include processor for further queries
        }
        
        processed_data.append(processed_entry)
        print(f"Processed {client_name}: {len(relevant_news)} relevant news articles found")
    
    return {
        'processed_clients': processed_data,
        'rag_processor': rag_processor
    }

if __name__ == "__main__":
    # Process client data with RAG
    result = process_client_data_with_rag()
    
    # Save processed data
    # Note: We can't save the rag_processor directly due to the model, so we'll save without it
    save_data = []
    for client in result['processed_clients']:
        save_client = {k: v for k, v in client.items() if k != 'rag_processor'}
        save_data.append(save_client)
    
    with open('processed_client_data_rag.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed data saved to processed_client_data_rag.json")
    print(f"Vector database and metadata saved for future use")
    
    # Print summary
    print(f"\n=== RAG PROCESSING SUMMARY ===")
    for client in result['processed_clients']:
        print(f"\n{client['client_name']}:")
        print(f"  Keywords: {', '.join(client['landing_page_keywords'][:3])}")
        print(f"  Relevant news: {len(client['relevant_news'])} articles")
        
        # Show top relevant news
        for i, news in enumerate(client['relevant_news'][:2]):
            print(f"    {i+1}. {news['title'][:60]}... (score: {news['similarity_score']:.3f})")