"""
RAG-Enabled NLP Processor for Alphix ML Challenge
Implements vector database and semantic search for news-responsive ad generation
"""
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import nltk

# Download NLTK stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from rake_nltk import Rake

class RAGProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG processor with embedding model and vector database
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.rake = Rake()
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.news_database = []
        self.client_database = []
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.model.encode(text)
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords using RAKE"""
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()[:max_keywords]
    
    def build_vector_database(self, client_data: List[Dict[str, Any]]) -> None:
        """
        Build FAISS vector database from client and news data
        
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
                # Chunk landing page content for better retrieval
                chunks = self._chunk_text(client['landing_page_content'], max_length=512)
                
                for i, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    embeddings.append(embedding)
                    metadata.append({
                        'type': 'landing_page',
                        'client_name': client['client_name'],
                        'url': client['url'],
                        'chunk_id': i,
                        'content': chunk,
                        'keywords': self.extract_keywords(chunk)
                    })
        
        # Process news articles
        for client in client_data:
            for article in client['news_articles']:
                # Combine title and source for richer embedding
                article_text = f"{article['title']} {article.get('source', '')}"
                embedding = self.get_embedding(article_text)
                embeddings.append(embedding)
                metadata.append({
                    'type': 'news_article',
                    'client_name': client['client_name'],
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
    
    def semantic_search(self, query: str, k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search in vector database
        
        Args:
            query: Search query
            k: Number of results to return
            filter_type: Optional filter by content type ('landing_page' or 'news_article')
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if self.index is None:
            raise ValueError("Vector database not built yet. Call build_vector_database() first.")
        
        # Get query embedding
        query_embedding = self.get_embedding(query).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            metadata = self.metadata[idx]
            
            # Apply filter if specified
            if filter_type and metadata['type'] != filter_type:
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
        Find relevant news articles for a specific client's landing page
        
        Args:
            client_name: Name of the client
            landing_page_content: Content of the landing page
            k: Number of relevant news articles to return
            
        Returns:
            List of relevant news articles with similarity scores
        """
        # Extract key themes from landing page
        keywords = self.extract_keywords(landing_page_content, max_keywords=10)
        query = f"{landing_page_content[:500]} {' '.join(keywords)}"
        
        # Search for relevant news articles
        results = self.semantic_search(query, k=k, filter_type='news_article')
        
        return results
    
    def get_contextual_information(self, client_name: str, topic: str, k: int = 5) -> Dict[str, Any]:
        """
        Get contextual information for ad generation
        
        Args:
            client_name: Name of the client
            topic: Topic or theme to focus on
            k: Number of results per category
            
        Returns:
            Dictionary with landing page context and relevant news
        """
        # Get landing page context
        landing_page_results = self.semantic_search(
            f"{client_name} {topic}", 
            k=k, 
            filter_type='landing_page'
        )
        
        # Get relevant news
        news_results = self.semantic_search(
            f"{topic} finance investment", 
            k=k, 
            filter_type='news_article'
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

def process_client_data_with_rag(client_data_file: str = 'client_data_with_content.json') -> Dict[str, Any]:
    """
    Process client data and build RAG system
    
    Args:
        client_data_file: Path to client data file
        
    Returns:
        Processed data with RAG capabilities
    """
    # Load client data
    with open(client_data_file, 'r', encoding='utf-8') as f:
        client_data = json.load(f)
    
    # Initialize RAG processor
    rag_processor = RAGProcessor()
    
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
        
        # Extract keywords from landing page
        keywords = rag_processor.extract_keywords(landing_page_content)
        
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