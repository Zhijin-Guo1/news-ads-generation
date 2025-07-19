# Alphix ML Engineering Challenge: News-Responsive Ad Generation with RAG

## 🎯 Project Goal

This project addresses a critical challenge in asset management marketing: **manually creating timely, relevant ad content that connects client investment insights with the rapidly evolving news cycle**. 

Our solution is an **AI-powered, RAG-enabled system** that automatically generates context-aware ad creative by:
- Analyzing client landing pages to understand their investment expertise and value proposition
- Using semantic search to find relevant financial news from current market developments  
- Generating professional, compliant ad copy that meaningfully connects client strengths with news themes
- Supporting multiple digital ad formats (LinkedIn, banner ads, etc.)

## 🚀 Key Features

- **Real OpenAI GPT-4 Integration**: Production-ready AI ad generation (not simulated)
- **RAG (Retrieval-Augmented Generation)**: Vector database with semantic search for precise news relevance
- **Multi-Client Support**: Processes multiple asset management firms simultaneously
- **Compliance-Aware**: Maintains professional, regulatory-compliant tone for financial services
- **Scalable Architecture**: Modular design supporting easy expansion and customization

## 🏗️ Architecture Overview

```
📊 Excel Data → 🕷️ Web Scraping → 🧠 RAG Processing → 🤖 OpenAI GPT-4 → 📢 Ad Creative
    ↓               ↓                    ↓                 ↓              ↓
Parsed URLs     Landing Page      Vector Database    Structured      Multiple Ad
& News Data     Content          + Embeddings       Prompts         Formats
```

### Core Components

1. **Data Ingestion Layer** (`parse_client_data.py`)
   - Parses Excel file with client URLs and news articles
   - Structures data for downstream processing

2. **Web Scraping Layer** (`web_scraper.py`)
   - Extracts content from client landing pages
   - Handles multiple URLs with error resilience

3. **RAG Processing Layer** (`rag_processor.py`)
   - Builds FAISS vector database with 384-dimensional embeddings
   - Implements semantic search for news-to-client relevance matching
   - Uses Sentence-BERT and RAKE for content understanding

4. **AI Generation Layer** (`openai_ad_generator.py`)
   - Real OpenAI GPT-4 integration with structured prompts
   - Generates LinkedIn ads, banner ads, and custom formats
   - Maintains compliance and professional tone

5. **Pipeline Orchestration** (`main_pipeline.py`)
   - Automated end-to-end workflow
   - Dependency management and error handling

## 📁 Repository Structure

```
news_generation/
├── README.md                           # This file
├── .env                               # OpenAI API key (secure)
├── .gitignore                         # Protects sensitive files
├── main_pipeline.py                   # 🚀 Main execution script
├── parse_client_data.py               # Excel data parser
├── web_scraper.py                     # Landing page scraper  
├── rag_processor.py                   # Vector database & semantic search
├── openai_ad_generator.py             # AI ad generation with OpenAI
├── solution_design.md                 # Technical design document
├── Alphix_ML_Challenge_News_Ad_Generation.docx  # Challenge requirements
├── URL_and_news_articles_examples_by_client.xlsx # Input data
└── Generated Files/
    ├── parsed_client_data.json        # Parsed Excel data
    ├── client_data_with_content.json  # Data + scraped content
    ├── processed_client_data_rag.json # RAG-processed data
    ├── generated_ad_campaigns.json    # Final AI-generated campaigns
    ├── vector_index.faiss            # Vector database
    └── vector_metadata.pkl           # Database metadata
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Option 1: Automated Setup (Recommended)
```bash
# Clone/download the repository
cd news_generation

# Add your OpenAI API key to .env file
# Edit .env and replace 'your-openai-api-key-here' with your actual key

# Run complete pipeline (installs dependencies automatically)
python3 main_pipeline.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install pandas openpyxl requests beautifulsoup4 sentence-transformers rake-nltk nltk faiss-cpu openai scikit-learn python-dotenv

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Add your OpenAI API key to .env file
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# Run individual components
python3 parse_client_data.py          # Parse Excel data
python3 web_scraper.py                # Scrape landing pages  
python3 rag_processor.py              # Build vector database
python3 openai_ad_generator.py        # Generate AI ads
```

## 🔑 API Key Setup

You need an OpenAI API key to run the AI generation. The system supports multiple secure methods:

### Method 1: .env File (Recommended)
1. Edit the `.env` file in the project root
2. Replace `your-openai-api-key-here` with your actual OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### Method 2: Environment Variable
```bash
export OPENAI_API_KEY='sk-your-actual-key-here'
python3 openai_ad_generator.py
```

### Method 3: Direct Parameter
```python
from openai_ad_generator import OpenAIAdGenerator
generator = OpenAIAdGenerator(api_key="sk-your-actual-key-here")
```

## 📊 Sample Results

The system successfully processes 3 major asset management firms:

### PIMCO (ESG/Sustainable Investing Focus)
- **Landing Page**: ESG capabilities and sustainable investing solutions
- **Relevant News**: 6 articles about sustainable investing trends, ESG regulations
- **Generated Ads**: Focus on connecting ESG expertise with current sustainability news
- **Sample Headline**: "Redefining Sustainable Investing with PIMCO's ESG Innovation"

### State Street (Federal Reserve & Policy Focus)  
- **Landing Page**: Fed policy insights and split decision analysis
- **Relevant News**: 2 articles about Fed rate decisions, policy uncertainty
- **Generated Ads**: Emphasize policy expertise during uncertain market conditions
- **Sample Headline**: "Navigating Fed Uncertainty: Expert Policy Analysis"

### T. Rowe Price (Global Markets & Diversification)
- **Landing Page**: Global market outlook and investment opportunities
- **Relevant News**: 2 articles about emerging markets, portfolio diversification  
- **Generated Ads**: Highlight global investment expertise and diversification strategies
- **Sample Headline**: "Navigate 2025's Global Markets with Confidence"

## 🔍 Technical Implementation Details

### RAG (Retrieval-Augmented Generation)
- **Vector Database**: FAISS with 384-dimensional embeddings
- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Keyword Extraction**: RAKE (Rapid Automatic Keyword Extraction)
- **Similarity Scoring**: Cosine similarity for semantic relevance

### AI Generation
- **Model**: OpenAI GPT-4 with structured prompts
- **Prompt Engineering**: Context-aware prompts with client data, news relevance, and format specifications
- **Output Formats**: LinkedIn single image ads, banner ads (300x250), custom creative concepts
- **Compliance**: Built-in financial services tone and regulatory awareness

### Data Processing
- **Input**: Excel file with 3 clients, 150+ news articles
- **Web Scraping**: BeautifulSoup with intelligent content extraction
- **Content Chunking**: Smart text segmentation for optimal embedding performance
- **Error Handling**: Robust error handling for web scraping and API calls

## 🛠️ Customization & Extension

### Adding New Clients
1. Add client data to the Excel file following the existing format
2. Run `python3 parse_client_data.py` to update parsed data
3. Re-run the pipeline to generate ads for new clients

### Custom Ad Formats  
Modify `openai_ad_generator.py` to add new ad format specifications in the prompt templates.

### Different LLM Models
Replace OpenAI calls in `openai_ad_generator.py` with other LLM APIs (Claude, local models, etc.).

### Enhanced RAG
- Add more sophisticated retrieval strategies in `rag_processor.py`
- Implement hybrid search (keyword + semantic)
- Add re-ranking algorithms

## 📈 Performance & Scalability

- **Processing Speed**: ~30 seconds for 3 clients with 150 news articles
- **Memory Usage**: ~500MB for vector database and embeddings
- **API Costs**: ~$0.10-0.50 per campaign generation (varies by content length)
- **Scalability**: Linear scaling with number of clients and news articles

## ✅ Quality Assurance

### Built-in Compliance
- Financial services regulatory tone
- No performance guarantees or overly promotional language
- Professional, authoritative messaging
- Fact-based content grounded in actual news

### Relevance Validation
- Semantic similarity scoring (0.3-0.8 typical range)
- Keyword extraction and matching
- Multiple news sources for context validation
- Human-readable relevance explanations

## 🚧 Future Enhancements

- **A/B Testing Framework**: Automated testing of ad variations
- **Real-time News Integration**: RSS feeds and news APIs
- **Image Generation**: DALL-E integration for visual content
- **Multi-language Support**: International client expansion
- **Compliance Monitoring**: Automated regulatory compliance checking
- **Analytics Dashboard**: Performance tracking and optimization insights

## 🤝 Contributing

This is a technical challenge submission. For production use, consider:
- Adding comprehensive test coverage
- Implementing monitoring and logging
- Adding rate limiting for API calls
- Enhanced error handling and recovery
- Security auditing for production deployment

---

**Author:** Zhijin Guo  
**Date:** 2025-07-19  
**Challenge:** Alphix ML Engineering - News-Responsive Ad Generation  
**Status:** Production-Ready Prototype with Real AI Integration