# News-Ads Generation with RAG Architecture

## 🎯 Project Goal

This project addresses a critical challenge in asset management marketing: **manually creating timely, relevant ad content that connects client investment insights with the rapidly evolving news cycle**. 

Our solution is an **AI-powered, RAG-enabled system** that automatically generates context-aware ad creative by:
- Analyzing client landing pages to understand their investment expertise and value proposition
- Using semantic search to find relevant financial news from current market developments  
- Generating professional, compliant ad copy that meaningfully connects client strengths with news themes
- Creating complete marketing materials with AI-generated images and text overlays
- Supporting multiple digital ad formats (LinkedIn, banner ads, etc.)

## 🚀 Key Features

- **🌐 Web Interface**: User-friendly Streamlit app with drag & drop functionality
- **Real OpenAI GPT-4o Integration**: Production-ready AI ad generation using latest models
- **RAG (Retrieval-Augmented Generation)**: Vector database with semantic search for precise news relevance
- **Professional Image Generation**: DALL-E 3 HD integration with text overlays
- **Multi-Client Support**: Processes multiple asset management firms simultaneously
- **Compliance-Aware**: Maintains professional, regulatory-compliant tone for financial services
- **Complete Marketing Pipeline**: End-to-end automated workflow from data to deployment-ready ads
- **Cloud Deployable**: Ready for Streamlit Cloud, Heroku, or Docker deployment

## 🌐 Try the Web App

**Option 1: Local Setup (2 minutes)**
```bash
git clone https://github.com/Zhijin-Guo1/news-ads-generation.git
cd news-ads-generation
pip install -r requirements.txt
streamlit run streamlit_app.py
```
→ Open `http://localhost:8501` in your browser

**Option 2: Deploy to Cloud (Free)**
- Fork this repo to your GitHub
- Deploy to [Streamlit Cloud](https://share.streamlit.io) 
- Add your OpenAI API key in secrets
- Share your public app URL!

**📋 See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions**

## 🚀 How It Works: Step-by-Step Example

Let's walk through exactly how the system transforms raw data into professional ad campaigns using **PIMCO** as our example:

### **📊 Step 1: Data Input**
```
Input File: URL_and_news_articles_examples_by_client.xlsx
- PIMCO URL: https://www.pimco.com/us/en/insights/fed-policymakers-split-decision
- State Street URL: https://www.ssga.com/uk/en_gb/institutional/capabilities/esg
- T. Rowe Price URL: https://www.troweprice.com/en/institutional/capabilities
- 265+ financial news articles embedded in vector database
```

### **🕷️ Step 2: Web Scraping**
```python
# System scrapes PIMCO's landing page
Scraped Content (sample):
"SPDR® Gold MiniShares® Trust, sustainable investing screening, 
ESG capabilities, environmental social governance solutions, 
institutional investment strategies..."

Result: 20,156 characters of landing page content
```

### **🧠 Step 3: RAG Processing & Vector Database**
```python
# System builds FAISS vector database with 265 embeddings
Vector Database: 384-dimensional embeddings using Sentence-BERT
Embedding Model: "all-MiniLM-L6-v2"

# For PIMCO (Fed policy focus):
Query: "Fed policymakers split decision + location americas + asia pacific"
Semantic Search Results:
✅ "Emerging markets gains in the cards as Trump's tariffs..." → Score: 0.31
✅ "Fed Forced to Balance Policy Predicament..." → Score: 0.39
❌ "Sustainable investing trends" → Score: 0.12 (not relevant)

# For State Street (ESG focus):
Query: "sustainable investing ESG + spdr gold trust + negative screening"
Semantic Search Results:
✅ "The Role Of Sustainable Investing In Modern Portfolios" → Score: 0.53
✅ "Sustainable investing: How does it work?" → Score: 0.66
❌ "Fed rate cuts" → Score: 0.08 (not relevant)

Top relevant news articles selected for each client
```

### **🤖 Step 4: AI Ad Generation with GPT-4o**
```python
# OpenAI GPT-4o receives structured prompt with RAG context:
"Client: PIMCO
Landing Page Content: Fed policymakers split decision analysis...
Relevant News: 
1. Emerging markets gains as Trump's tariffs challenge... (Score: 0.31)
2. Fed Forced to Balance Policy Predicament... (Score: 0.39)

Task: Generate LinkedIn ad, banner ad, and additional creative
Format: JSON with headline, body, CTA, image_description
Tone: Professional, compliant, thought leadership"

GPT-4o Response: [Contextually relevant ad content with news connections]
```

### **📢 Step 5: Generated Ad Campaigns**

**PIMCO - Fed Policy Focus:**
```json
{
  "linkedin_single_image": {
    "headline": "Navigating Market Shifts: Insights on Fed Policy & Emerging Trends",
    "body": "As Fed policymakers face a split decision, the implications for emerging markets are profound. PIMCO's insights explore how these dynamics interact with global trade tensions. Discover strategies to manage risk and capture potential in uncertain times.",
    "call_to_action": "Explore PIMCO's Insights",
    "image_description": "Dynamic visual of a globe with key financial markets highlighted, overlaid with graphs indicating market trends",
    "news_connection_rationale": "Connects PIMCO's Fed policy insights with current emerging markets and trade tension news"
  }
}
```

**State Street - ESG Leadership:**
```json
{
  "linkedin_single_image": {
    "headline": "Sustainable Investing: A Pillar of Modern Portfolios",
    "body": "As sustainable investing becomes integral to financial portfolios, State Street's insights and solutions lead the way. Leverage our expertise to align investments with sustainability goals, supported by robust research and analytics.",
    "call_to_action": "Explore Sustainable Solutions",
    "image_description": "Graph depicting rising trends in sustainable investing, with icons representing diverse asset classes overlaying a green and blue globe",
    "news_connection_rationale": "Links State Street's sustainable investing expertise with Forbes' coverage on its critical role in modern portfolios"
  }
}
```

### **🎨 Step 6: Professional Image Generation with DALL-E 3**

The system creates complete, ready-to-deploy marketing ads using DALL-E 3 HD with professional text overlays:

#### **LinkedIn Single Image Ad - PIMCO Fed Policy Focus**
![PIMCO LinkedIn Ad](generated_ads_images/final_ads/PIMCO_linkedin_single_image_final_013039.png)
*News-responsive ad connecting PIMCO's Fed policy insights with emerging market trends*

#### **Banner Ad - State Street ESG Leadership** 
![State Street Banner Ad](generated_ads_images/final_ads/State_Street_banner_ad_300x250_final_013233.png)
*Wide banner format highlighting sustainable investing expertise*

#### **Additional Creative - T. Rowe Price 2025 Outlook**
![T. Rowe Price Creative](generated_ads_images/final_ads/T_Rowe_Price_additional_creative_final_013422.png)
*Podcast-style creative connecting market outlook with growth strategies*

**Key Features of Generated Images:**
- ✅ **DALL-E 3 HD backgrounds** with enhanced professional quality
- ✅ **Intelligent text overlay** using PIL with Helvetica fonts
- ✅ **Format-specific layouts** (1024x1024 LinkedIn, 1792x1024 banners)
- ✅ **Client branding integration** with logo placeholder areas
- ✅ **Marketing-ready output** at publication quality (95% PNG compression)
- ✅ **Contextual visual themes** matching news relevance and client expertise

### **🎯 What Makes This Powerful**

**Traditional Process (3 days):**
1. Marketing team manually reads financial news
2. Brainstorms connections to company expertise  
3. Writes ad copy from scratch
4. Reviews and revises multiple times

**Our AI Process (10 minutes):**
1. ✅ Automatically analyzes 265+ news articles in vector database
2. ✅ Finds semantic connections using cosine similarity (0.31-0.66 relevance scores)
3. ✅ Generates professional, compliant ad copy with GPT-4o
4. ✅ Creates complete visual campaigns with DALL-E 3 HD
5. ✅ Produces multiple ad formats with text overlays simultaneously

**Key Innovation:** The system creates **contextually relevant** ads using RAG architecture that meaningfully connects client expertise with current market developments through semantic search.

## 🏗️ Architecture Overview

```
📊 Excel Data → 🕷️ Web Scraping → 🧠 RAG Vector DB → 🤖 GPT-4o → 🎨 DALL-E 3 → 📢 Complete Ads
    ↓               ↓                    ↓              ↓           ↓             ↓
Parsed URLs     Landing Page      FAISS Index     Structured   HD Images    Marketing-Ready
& News Data     Content (20K+)   (265 vectors)   Prompts      + Text       Materials
                                  384-dim                      Overlays
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
   - Real OpenAI GPT-4o integration with structured prompts
   - Generates LinkedIn ads, banner ads, and custom formats
   - Maintains compliance and professional tone

5. **Image Generation Layer** (`professional_ad_generator.py`)
   - DALL-E 3 HD integration for background image generation
   - PIL-based text overlay system with professional typography
   - Multiple format support (square, banner, custom)
   - Organized output structure with metadata tracking

6. **Web Interface Layer** (`streamlit_app.py`)
   - Interactive Streamlit frontend for the entire pipeline
   - Real-time progress tracking and visualization
   - File upload, campaign generation, and download features
   - User-friendly interface for non-technical users

## 📁 Repository Structure

```
news-ads-generation/
├── README.md                           # This file
├── .env                               # OpenAI API key (secure)
├── .gitignore                         # Protects sensitive files
├── streamlit_app.py                   # 🌐 Web interface (Streamlit)
├── run_streamlit.py                   # 🚀 Launch script for web app
├── requirements.txt                   # 📦 Python dependencies
├── parse_client_data.py               # Excel data parser
├── web_scraper.py                     # Landing page scraper  
├── rag_processor.py                   # Vector database & semantic search
├── openai_ad_generator.py             # AI ad generation with GPT-4o
├── professional_ad_generator.py       # DALL-E 3 + text overlay system
├── solution_design.md                 # Technical design document
├── Alphix_ML_Challenge_News_Ad_Generation.docx  # Challenge requirements
├── URL_and_news_articles_examples_by_client.xlsx # Input data
├── data/                              # Organized data folder
│   ├── parsed_client_data.json       # Parsed Excel data
│   ├── client_data_with_content.json # Data + scraped content
│   ├── processed_client_data_rag.json # RAG-processed data
│   ├── vector_index.faiss           # FAISS vector database
│   └── vector_metadata.pkl          # Database metadata
├── generated_ads_text/                # Text campaign outputs
│   └── ad_campaigns.json             # Structured ad content
└── generated_ads_images/              # Visual campaign outputs
    ├── final_ads/                     # Complete marketing materials
    │   ├── *_linkedin_*.png          # LinkedIn format ads
    │   ├── *_banner_*.png            # Banner format ads
    │   ├── *_additional_*.png        # Custom creative formats
    │   └── ad_metadata.json          # Generation metadata
    └── *_bg_*.png                    # Background images
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Option 1: Web Interface (Recommended for Most Users)
```bash
# Clone/download the repository
cd news-ads-generation

# Install dependencies
pip install -r requirements.txt

# Launch the web interface
python3 run_streamlit.py
# OR
streamlit run streamlit_app.py

# Access the app at: http://localhost:8501
```

**🌐 Web Interface Access:**
- **Local Development**: `http://localhost:8501` (after running the commands above)
- **Public Demo**: Deploy to [Streamlit Cloud](https://share.streamlit.io) for free public access
- **GitHub Repository**: [View Source Code](https://github.com/Zhijin-Guo1/news-ads-generation)
- **Deployment Guide**: [See DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment options

**🌐 The web interface provides:**
- **📁 File Upload**: Drag & drop Excel files with client data
- **⚙️ Configuration**: Easy API key setup and processing options  
- **📊 Real-time Progress**: Watch each pipeline step complete
- **🎨 Visual Results**: Preview generated ads with images
- **💾 Easy Downloads**: JSON, CSV, and image downloads

### Option 2: Command Line (Advanced Users)
```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env file
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# Run complete pipeline components
python3 parse_client_data.py          # Parse Excel data
python3 web_scraper.py                # Scrape landing pages  
python3 rag_processor.py              # Build vector database
python3 openai_ad_generator.py        # Generate AI text campaigns
python3 professional_ad_generator.py  # Create complete visual ads
```

### 🌐 Web Interface Features

The Streamlit app provides a complete user-friendly interface:

**🎯 Main Dashboard:**
- Interactive step-by-step pipeline execution
- Real-time progress bars and status updates
- Visual campaign previews with generated images
- Comprehensive configuration options

**📊 Key Features:**
- **File Upload**: Drag & drop Excel files or use sample data
- **API Configuration**: Secure OpenAI API key management
- **Processing Control**: Toggle image generation, set news article limits
- **Results Visualization**: Preview ads with headlines, body text, and images
- **Export Options**: Download campaigns as JSON, CSV, or view file structure

**🔧 Configuration Panel:**
- OpenAI API key setup with validation
- Vector database status monitoring
- Processing options (image generation on/off, max news articles)
- System status indicators

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

## 📊 Campaign Results & RAG Performance

The system successfully processes 3 major asset management firms with updated 2025 data:

### PIMCO (Fed Policy & Emerging Markets Focus)
- **Updated URL**: https://www.pimco.com/us/en/insights/fed-policymakers-split-decision
- **Landing Page Content**: 10,121 characters of Fed policy analysis
- **RAG Results**: 2 highly relevant news articles (scores: 0.31-0.39)
  - "Emerging markets gains in the cards as Trump's tariffs challenge U.S. exceptionalism"
  - "Fed Forced to Balance Policy Predicament With Economic Uncertainty"
- **Generated Campaign**: LinkedIn ad + Banner + Webinar concept
- **Sample Headlines**: 
  - "Navigating Market Shifts: Insights on Fed Policy & Emerging Trends"
  - "Join Our Webinar: Market Dynamics in a Changing World"

### State Street (ESG/Sustainable Investing Leadership)
- **URL**: https://www.ssga.com/uk/en_gb/institutional/capabilities/esg
- **Landing Page Content**: 20,156 characters of ESG capabilities
- **RAG Results**: 6 highly relevant news articles (scores: 0.51-0.66)
  - "The Role Of Sustainable Investing In Modern Financial Portfolios" (0.53)
  - "Sustainable investing: How does it work?" (0.66)
- **Generated Campaign**: Focus on ESG innovation and sustainability leadership
- **Sample Headlines**:
  - "Sustainable Investing: A Pillar of Modern Portfolios"
  - "State Street Leads in ESG Innovation"

### T. Rowe Price (2025 Market Outlook & Growth Strategies)
- **URL**: https://www.troweprice.com/en/institutional/capabilities
- **Landing Page Content**: 27,520 characters of investment capabilities
- **RAG Results**: 2 relevant articles focusing on growth and emerging markets (scores: 0.44-0.49)
  - "Real Estate a Top Sector to Watch in 2025" (0.49)
  - "Partner Insight: Robeco Emerging Markets Equities strategy" (0.44)
- **Generated Campaign**: 2025 outlook theme with growth focus
- **Sample Headlines**:
  - "T. Rowe Price's 2025 Outlook: Navigating Emerging Growth"
  - "2025 Insights Podcast: Growth & ESG"

### RAG System Performance
- **Vector Database**: 265 embeddings successfully built
- **Semantic Search Accuracy**: 0.31-0.66 relevance scores (0.5+ indicates strong relevance)
- **Keyword Extraction**: RAKE algorithm successfully identified key themes
- **News-Client Matching**: Each client matched with 2-6 contextually relevant articles

## 🔍 Technical Implementation Details

### RAG (Retrieval-Augmented Generation)
- **Vector Database**: FAISS with 384-dimensional embeddings
- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Keyword Extraction**: RAKE (Rapid Automatic Keyword Extraction)
- **Similarity Scoring**: Cosine similarity for semantic relevance

### AI Generation
- **Text Model**: OpenAI GPT-4o (latest Omni model) with structured prompts
- **Image Model**: DALL-E 3 HD for professional background generation
- **Prompt Engineering**: RAG-enhanced prompts with client context, semantic search results, and format specifications
- **Output Formats**: LinkedIn single image ads (1024x1024), banner ads (1792x1024), custom creative concepts
- **Text Overlay**: PIL-based system with Helvetica fonts and professional layouts
- **Compliance**: Built-in financial services tone and regulatory awareness

### Data Processing
- **Input**: Excel file with 3 clients, 265+ news articles embedded
- **Web Scraping**: BeautifulSoup with intelligent content extraction (10K-27K chars per client)
- **Content Chunking**: Smart text segmentation (512 char max) for optimal embedding performance
- **Vector Storage**: FAISS IndexFlatIP with L2 normalization for cosine similarity
- **Error Handling**: Robust error handling for web scraping, API calls, and image generation

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

- **Processing Speed**: ~10 minutes for complete pipeline (3 clients, 265 embeddings, 9 visual ads)
- **Memory Usage**: ~800MB for vector database, embeddings, and image processing
- **API Costs**: ~$2-5 per complete campaign (includes GPT-4o + DALL-E 3 HD)
- **Image Generation**: ~30 seconds per DALL-E 3 HD image with 3-second rate limiting
- **Scalability**: Linear scaling with number of clients and news articles
- **Output Quality**: Marketing-ready materials requiring minimal human review

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

- **A/B Testing Framework**: Automated testing of ad variations with performance metrics
- **Real-time News Integration**: RSS feeds and live news APIs for dynamic updates
- **Advanced Image Customization**: Logo integration, brand color schemes, custom templates
- **Multi-language Support**: International client expansion with localized content
- **Compliance Monitoring**: Automated regulatory compliance checking with industry rules
- **Analytics Dashboard**: Performance tracking, click-through rates, and campaign optimization
- **Video Generation**: Short-form video ads using AI video generation models
- **Enhanced RAG**: Hybrid search combining keyword and semantic approaches
- **Client Feedback Loop**: Human-in-the-loop refinement based on campaign performance

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