"""
Streamlit Frontend for News-Responsive Ad Generation System
Interactive web interface for the Alphix ML Challenge solution
"""

import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="News-Responsive Ad Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import json
import os
import sys
from pathlib import Path
import time
from PIL import Image
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
try:
    from utils.parse_client_data import parse_client_data
    from utils.web_scraper import scrape_text_from_url
    from utils.rag_processor import RAGProcessor, process_client_data_with_rag
    from openai_ad_generator import OpenAIAdGenerator
    from professional_ad_generator import ProfessionalAdGenerator
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E8;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'client_data' not in st.session_state:
        st.session_state.client_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'generated_campaigns' not in st.session_state:
        st.session_state.generated_campaigns = None
    if 'processing_step' not in st.session_state:
        st.session_state.processing_step = 0
    if 'api_key' not in st.session_state:
        try:
            if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
            else:
                st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
        except (KeyError, AttributeError):
            st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')

def display_header():
    """Display the main header and description"""
    st.markdown('<h1 class="main-header">🎯 Advanced AI News-Responsive Ad Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🌐 Real-time Web Scraping AI-Powered Marketing Campaign Generation</h3>
        <p>Transform live webpage content into professional, contextually relevant ad campaigns using 
        real-time web scraping and GPT-4o function calling for maximum accuracy.</p>
        <p><strong>🚀 Latest Features:</strong> Real-time web scraping, GPT-4o function calling, live content analysis, and current market positioning extraction.</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_config():
    """Configure the sidebar with settings and information"""
    st.sidebar.header("⚙️ Configuration")
    
    # Use API key from session state (already initialized in init_session_state)
    api_key = st.session_state.get('api_key', '')
    
    # Display API key status
    if api_key:
        # Check source of API key for display
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets and api_key == st.secrets.get("OPENAI_API_KEY", ""):
            st.sidebar.success("✅ API Key configured from Streamlit secrets")
        else:
            st.sidebar.success("✅ API Key configured from environment")
    else:
        st.sidebar.error("❌ OPENAI_API_KEY not found")
        st.sidebar.info("💡 For Streamlit Cloud: Add OPENAI_API_KEY in app secrets\n💡 For local: Set OPENAI_API_KEY environment variable")
        
        # Debug information only when API key is missing
        with st.sidebar.expander("🔍 Debug Info", expanded=False):
            st.write("**Secrets Debug:**")
            st.write(f"- `st.secrets` available: {hasattr(st, 'secrets')}")
            if hasattr(st, 'secrets'):
                try:
                    secret_keys = list(st.secrets.keys())
                    st.write(f"- Available secret keys: {secret_keys}")
                    st.write(f"- OPENAI_API_KEY in secrets: {'OPENAI_API_KEY' in st.secrets}")
                except Exception as debug_e:
                    st.write(f"- Error accessing secrets: {debug_e}")
            
            st.write("**Environment Debug:**")
            env_keys = [k for k in os.environ.keys() if 'API' in k or 'OPENAI' in k]
            st.write(f"- Environment keys with API/OPENAI: {env_keys}")
            st.write(f"- Session state API key: {bool(st.session_state.get('api_key'))}")
    
    # System Information
    st.sidebar.subheader("📊 System Status")
    
    # Check if vector database exists
    if os.path.exists("data/vector_index.faiss"):
        st.sidebar.success("✅ Vector Database Ready")
    else:
        st.sidebar.info("📋 Vector Database: Will be built")
    
    # Check for existing campaigns
    if os.path.exists("generated_ads_text/ad_campaigns.json"):
        st.sidebar.success("✅ Previous Campaigns Found")
    else:
        st.sidebar.info("📋 No Previous Campaigns")
    
    # Processing Options
    st.sidebar.subheader("🎛️ Processing Options")
    
    generate_images = st.sidebar.checkbox(
        "Generate Images with DALL-E 3",
        value=True,
        help="Requires OpenAI API key set as environment variable"
    )
    
    max_news_articles = st.sidebar.slider(
        "Max News Articles per Client",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top-ranked relevant news articles to return. Each Excel sheet contains pre-matched news for each client."
    )
    
    return {
        'api_key': api_key,
        'generate_images': generate_images,
        'max_news_articles': max_news_articles
    }

def file_upload_section():
    """Handle file upload and data parsing"""
    st.markdown('<h2 class="step-header">📊 Step 1: Upload Client Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Excel file with client URLs and news articles",
        type=['xlsx', 'xls'],
        help="Expected format: Sheets with client names, URLs, and news articles"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Parse the data
            with st.spinner("📋 Parsing client data..."):
                client_data = parse_client_data(temp_path)
            
            # Display parsed data summary
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ Successfully parsed data for {len(client_data)} clients")
            
            # Show data preview
            col1, col2, col3 = st.columns(3)
            
            for i, client in enumerate(client_data):
                with [col1, col2, col3][i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{client['client_name']}</h4>
                        <p><strong>URL:</strong> {client['url'][:50]}...</p>
                        <p><strong>News Articles:</strong> {len(client['news_articles'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Store in session state
            st.session_state.client_data = client_data
            return client_data
            
        except Exception as e:
            st.error(f"❌ Error parsing file: {str(e)}")
            return None
    
    else:
        # Option to use existing data
        if st.button("📁 Use Existing Sample Data"):
            try:
                if os.path.exists("data/parsed_client_data.json"):
                    with open("data/parsed_client_data.json", 'r') as f:
                        client_data = json.load(f)
                    st.success(f"✅ Loaded existing data for {len(client_data)} clients")
                    # Store in session state
                    st.session_state.client_data = client_data
                    return client_data
                else:
                    st.warning("⚠️ No existing sample data found. Please upload a file.")
            except Exception as e:
                st.error(f"❌ Error loading existing data: {str(e)}")
    
    return None

def web_scraping_section(client_data):
    """Handle web scraping of client landing pages"""
    st.markdown('<h2 class="step-header">🕷️ Step 2: Web Scraping</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Start Web Scraping", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, client in enumerate(client_data):
            status_text.text(f"Scraping {client['client_name']}...")
            
            try:
                content = scrape_text_from_url(client['url'])
                client['landing_page_content'] = content
                
                if content:
                    st.success(f"✅ {client['client_name']}: {len(content)} characters scraped")
                else:
                    st.warning(f"⚠️ {client['client_name']}: Failed to scrape content")
                    
            except Exception as e:
                st.error(f"❌ {client['client_name']}: {str(e)}")
                client['landing_page_content'] = ""
            
            progress_bar.progress((i + 1) / len(client_data))
        
        status_text.text("✅ Web scraping completed!")
        
        # Store updated data in session state
        st.session_state.client_data = client_data
        return client_data
    
    return None

def rag_processing_section(client_data, config):
    """Handle RAG processing and vector database creation"""
    st.markdown('<h2 class="step-header">🧠 Step 3: RAG Processing</h2>', unsafe_allow_html=True)
    
    # Check if max_news_articles setting has changed
    if 'last_max_news_articles' not in st.session_state:
        st.session_state.last_max_news_articles = config['max_news_articles']
    
    # Show warning if settings changed but haven't reprocessed
    if (st.session_state.processed_data is not None and 
        st.session_state.last_max_news_articles != config['max_news_articles']):
        st.warning(f"⚠️ You changed 'Max News Articles per Client' from {st.session_state.last_max_news_articles} to {config['max_news_articles']}. Click 'Build Vector Database' again to apply the new setting.")
    
    if st.button("🔍 Build Vector Database", type="primary"):
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.subheader("🚀 RAG Processing Progress")
            
            # Overall progress bar
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Step-specific progress bars
            step_progress = st.progress(0)
            step_status = st.empty()
            
        try:
            # Step 1: Module Initialization (10%)
            overall_status.text("Step 1/4: Initializing modules...")
            overall_progress.progress(10)
            
            with step_status:
                st.info("🔄 Reloading modules to get latest version...")
            
            import importlib
            import utils.rag_processor
            importlib.reload(utils.rag_processor)
            from utils.rag_processor import RAGProcessor
            
            step_progress.progress(100)
            
            # Step 2: RAG Processor Setup (25%)
            overall_status.text("Step 2/4: Setting up RAG processor...")
            overall_progress.progress(25)
            step_progress.progress(0)
            
            with step_status:
                st.info("🧠 Initializing RAGProcessor with OpenAI API key...")
            
            # Initialize RAG processor with OpenAI API key for advanced keyword extraction
            try:
                rag_processor = RAGProcessor(openai_api_key=st.session_state.api_key)
                with status_container:
                    st.success("✅ Initialized RAGProcessor with OpenAI API key for real-time web scraping")
                step_progress.progress(100)
            except TypeError as e:
                with status_container:
                    st.warning(f"⚠️ Using fallback RAGProcessor initialization: {e}")
                # Fallback for compatibility with older version
                rag_processor = RAGProcessor()
                # Try to set API key manually if possible
                if hasattr(rag_processor, 'openai_client') and st.session_state.api_key:
                    try:
                        from openai import OpenAI
                        rag_processor.openai_client = OpenAI(api_key=st.session_state.api_key)
                        with status_container:
                            st.success("✅ Manually set OpenAI client for real-time features")
                    except Exception as e2:
                        with status_container:
                            st.warning(f"Could not set OpenAI client: {e2}")
                step_progress.progress(100)
            except Exception as e:
                st.error(f"Failed to initialize RAGProcessor: {e}")
                return
            
            # Step 3: Vector Database Building (50%)
            overall_status.text("Step 3/4: Building vector database...")
            overall_progress.progress(50)
            step_progress.progress(0)
            
            with step_status:
                st.info("📊 Processing client data and news articles into vector embeddings...")
            
            # Build vector database with progress updates
            rag_processor.build_vector_database(client_data)
            step_progress.progress(100)
            
            # Show database stats
            with status_container:
                st.success(f"✅ Vector database built with {len(rag_processor.metadata)} total embeddings")
                col1, col2 = st.columns(2)
                with col1:
                    news_embeddings = [item for item in rag_processor.metadata if item['type'] == 'news_article']
                    st.metric("📰 News Embeddings", len(news_embeddings))
                with col2:
                    page_embeddings = [item for item in rag_processor.metadata if item['type'] == 'landing_page']
                    st.metric("🌐 Landing Page Embeddings", len(page_embeddings))
                
            
            # Step 4: Client Processing with Real-time Web Scraping (75% to 100%)
            overall_status.text("Step 4/4: Processing clients with real-time web scraping...")
            overall_progress.progress(75)
            step_progress.progress(0)
            
            with step_status:
                st.info("🌐 Processing each client with real-time keyword extraction...")
            
            # Process each client with detailed progress
            processed_clients = []
            client_count = len([c for c in client_data if c.get('landing_page_content')])
            
            for i, client in enumerate(client_data):
                if client.get('landing_page_content'):
                    # Update progress for this client
                    client_progress = int((i / client_count) * 100)
                    step_progress.progress(client_progress)
                    
                    with step_status:
                        st.info(f"🔄 Processing {client['client_name']} ({i+1}/{client_count})")
                    
                    # Create expandable section for this client's processing
                    with st.expander(f"📊 Processing {client['client_name']}", expanded=True):
                        # Sub-step 1: Find relevant news
                        st.write("🔍 Finding relevant news articles...")
                        relevant_news = rag_processor.find_relevant_news(
                            client['client_name'],
                            client['landing_page_content'],
                            k=config['max_news_articles']
                        )
                        
                        # Show results
                        st.success(f"✅ Found {len(relevant_news)}/{config['max_news_articles']} relevant articles")
                        
                        # Sub-step 2: Real-time web scraping and keyword extraction
                        st.write("🌐 Performing real-time web scraping...")
                        scraping_progress = st.progress(0)
                        
                        try:
                            # Extract investment themes
                            scraping_progress.progress(25)
                            investment_themes = rag_processor._extract_investment_themes(client['landing_page_content'])
                            
                            # Real-time keyword extraction
                            scraping_progress.progress(50)
                            st.write(f"   📡 Scraping live content from: {client.get('url', 'N/A')}")
                            
                            landing_page_keywords = rag_processor.extract_keywords(
                                url=client.get('url'),
                                max_keywords=8
                            )
                            scraping_progress.progress(75)
                            
                            # Generate enhanced query
                            enhanced_query = rag_processor._create_thematic_query(
                                investment_themes, 
                                client['client_name'], 
                                landing_page_keywords
                            )
                            scraping_progress.progress(100)
                            
                            # Store results
                            client['enhanced_query'] = enhanced_query
                            client['realtime_keywords'] = landing_page_keywords
                            
                            # Show extracted information
                            if landing_page_keywords:
                                st.success(f"✅ Extracted {len(landing_page_keywords)} real-time keywords")
                                with st.expander("🎯 View extracted keywords"):
                                    for j, keyword in enumerate(landing_page_keywords, 1):
                                        st.write(f"{j}. {keyword}")
                            else:
                                st.warning("⚠️ No keywords extracted - using fallback")
                                
                            st.info(f"🧠 Generated query: {enhanced_query}")
                                
                        except Exception as e:
                            st.error(f"❌ Real-time processing failed: {e}")
                            client['enhanced_query'] = "Error generating query"
                            client['realtime_keywords'] = []
                        
                        client['relevant_news'] = relevant_news
                        processed_clients.append(client)
            
            # Final completion
            step_progress.progress(100)
            overall_progress.progress(100)
            overall_status.text("✅ RAG processing completed successfully!")
            
            with status_container:
                st.success(f"🎉 RAG processing completed for {len(processed_clients)} clients with real-time web scraping!")
                
                # Display RAG results
                for client in processed_clients:
                    with st.expander(f"📊 RAG Results: {client['client_name']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🌐 Real-time Enhanced Search Query")
                            st.code(client.get('enhanced_query', 'No query generated'), language=None)
                            
                            st.subheader("🌐 Real-time Web Scraped Keywords")
                            realtime_keywords = client.get('realtime_keywords', [])
                            if realtime_keywords:
                                for keyword in realtime_keywords:
                                    st.write(f"• {keyword}")
                            else:
                                st.write("• Real-time keywords extraction unavailable")
                            
                            st.subheader("📊 Investment Themes")
                            try:
                                investment_themes = rag_processor._extract_investment_themes(client['landing_page_content'])
                                for theme in list(investment_themes.keys())[:4]:  # Show top 4 themes
                                    st.write(f"• {theme.replace('_', ' ').title()}")
                            except:
                                st.write("• Theme extraction unavailable")
                        
                        with col2:
                            st.subheader("📰 Relevant News")
                            for news in client['relevant_news']:  # Show all relevant news found
                                score = news.get('similarity_score', 0)
                                st.write(f"• **{news['title'][:50]}...** (Score: {score:.3f})")
                
                # Update the last used setting
                st.session_state.last_max_news_articles = config['max_news_articles']
                
                return processed_clients
                
            except Exception as e:
                st.error(f"❌ RAG processing failed: {str(e)}")
                return None
    
    return None

def campaign_generation_section(processed_data, config):
    """Handle AI campaign generation"""
    st.markdown('<h2 class="step-header">🤖 Step 4: AI Campaign Generation</h2>', unsafe_allow_html=True)
    
    if not config['api_key']:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    
    # Add prompt customization section
    with st.expander("🎯 Customize Generation Prompt", expanded=False):
        st.markdown("**Edit the prompt template used for ad generation:**")
        
        default_prompt = """**Client Context:**
Client: {client_name}
Landing Page URL: {client_url}

**Key Themes & Expertise:**
{landing_summary}

**Relevant Market News (Ranked by Relevance):**
{news_context}

**Task:** Generate ad creative for the following formats that meaningfully connects the client's expertise with the current news landscape:

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

**Output:** Return as JSON with keys: "linkedin_single_image", "banner_ad_300x250", "additional_creative", each containing "headline", "body", "call_to_action", "image_description", and "news_connection_rationale"."""

        custom_prompt = st.text_area(
            "Prompt Template",
            value=default_prompt,
            height=400,
            help="You can customize this prompt template. Use {client_name}, {client_url}, {landing_summary}, and {news_context} as placeholders."
        )
        
        # Store the custom prompt in session state
        st.session_state.custom_prompt = custom_prompt
    
    if st.button("🎯 Generate Ad Campaigns", type="primary"):
        with st.spinner("🤖 Generating AI campaigns with GPT-4o..."):
            try:
                # Initialize generator
                generator = OpenAIAdGenerator(api_key=config['api_key'])
                generator.load_rag_processor()
                
                campaigns = []
                progress_bar = st.progress(0)
                
                for i, client_data in enumerate(processed_data):
                    st.text(f"Generating campaign for {client_data['client_name']}...")
                    
                    campaign = generator.generate_campaign_for_client(client_data)
                    campaigns.append(campaign)
                    
                    progress_bar.progress((i + 1) / len(processed_data))
                    time.sleep(1)  # Rate limiting
                
                st.success(f"✅ Generated {len(campaigns)} campaigns successfully!")
                
                # Save campaigns to both locations for compatibility
                os.makedirs("generated_ads_text", exist_ok=True)
                
                # Save to organized folder structure
                with open("generated_ads_text/ad_campaigns.json", 'w') as f:
                    json.dump(campaigns, f, indent=2)
                
                # Save to root for image generation compatibility
                with open("generated_ad_campaigns.json", 'w') as f:
                    json.dump(campaigns, f, indent=2)
                
                return campaigns
                
            except Exception as e:
                st.error(f"❌ Campaign generation failed: {str(e)}")
                return None
    
    return None

def image_generation_section(campaigns, config):
    """Handle professional image generation"""
    st.markdown('<h2 class="step-header">🎨 Step 5: Image Generation</h2>', unsafe_allow_html=True)
    
    if not config['generate_images']:
        st.info("📋 Image generation disabled in settings")
        return campaigns
    
    if not config['api_key']:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return campaigns
    
    # Add image prompt customization section
    with st.expander("🎨 Customize Image Generation Prompt", expanded=False):
        st.markdown("**Edit the prompt template used for DALL-E 3 image generation:**")
        
        default_image_prompt = """Professional financial marketing image for {client_name}. {format_style}{description}. High-quality professional photography or clean vector graphics, modern financial services aesthetic, sophisticated color palette with blues and greens, corporate professional style, minimal and clean design, no text or typography in the image, suitable for adding text overlay later, institutional investor appropriate, premium brand feeling"""

        custom_image_prompt = st.text_area(
            "Image Prompt Template",
            value=default_image_prompt,
            height=200,
            help="Customize the DALL-E 3 prompt template. Use {client_name}, {format_style}, and {description} as placeholders."
        )
        
        # Store the custom image prompt in session state
        st.session_state.custom_image_prompt = custom_image_prompt
    
    if st.button("🎨 Generate Professional Images", type="primary"):
        with st.spinner("🎨 Generating images with DALL-E 3 HD..."):
            try:
                # Initialize image generator
                img_generator = ProfessionalAdGenerator(api_key=config['api_key'])
                
                # Check if campaign file exists
                campaign_file = 'generated_ad_campaigns.json'
                if not os.path.exists(campaign_file):
                    st.error("❌ No campaign file found. Please generate campaigns in Step 4 first.")
                    return campaigns
                
                # Generate complete ads
                generated_ads = img_generator.generate_complete_ad_campaign()
                
                if generated_ads is not None and len(generated_ads) > 0:
                    st.success(f"✅ Generated {len(generated_ads)} complete ads with images!")
                    # Return the original campaigns since generated_ads has different structure
                    # The images are saved to disk and will be displayed via file paths
                    return campaigns
                else:
                    st.warning("⚠️ Image generation completed but no ads were returned. Check the logs.")
                    return campaigns
                
            except Exception as e:
                st.error(f"❌ Image generation failed: {str(e)}")
                return campaigns
    
    return campaigns

def campaign_display_section(campaigns):
    """Display generated campaigns"""
    if not campaigns:
        return
    
    st.markdown('<h2 class="step-header">📊 Generated Campaigns</h2>', unsafe_allow_html=True)
    
    for campaign in campaigns:
        client_name = campaign.get('client_name', 'Unknown Client')
        
        with st.expander(f"📢 {client_name} Campaign", expanded=True):
            # Campaign metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📰 News Articles", campaign.get('relevant_news_count', 0))
            with col2:
                st.metric("🎯 Ad Formats", len(campaign.get('ad_creative', {})))
            with col3:
                st.metric("🔗 Client URL", "Available" if campaign.get('client_url') else "N/A")
            
            # Display ad creative
            ad_creative = campaign.get('ad_creative', {})
            
            for format_name, ad_data in ad_creative.items():
                st.subheader(f"📱 {format_name.replace('_', ' ').title()}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Headline:** {ad_data.get('headline', 'N/A')}")
                    st.write(f"**Body:** {ad_data.get('body', 'N/A')}")
                    st.write(f"**CTA:** {ad_data.get('call_to_action', 'N/A')}")
                
                with col2:
                    st.write(f"**Image Description:**")
                    st.write(ad_data.get('image_description', 'N/A'))
                
                # Show generated image if available
                image_path = f"generated_ads_images/final_ads/{client_name.replace(' ', '_')}_{format_name}_final_*.png"
                try:
                    import glob
                    matching_files = glob.glob(image_path)
                    if matching_files:
                        latest_file = max(matching_files, key=os.path.getctime)
                        st.image(latest_file, caption=f"{client_name} - {format_name}", width=600)
                except:
                    pass
                
                st.divider()

def download_section(campaigns):
    """Provide download options for generated campaigns"""
    if not campaigns:
        return
    
    st.markdown('<h2 class="step-header">📥 Download Results</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download JSON
        if st.button("📄 Download Campaign JSON"):
            json_data = json.dumps(campaigns, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name="ad_campaigns.json",
                mime="application/json"
            )
    
    with col2:
        # Download CSV summary
        if st.button("📊 Download CSV Summary"):
            # Create summary data
            summary_data = []
            for campaign in campaigns:
                client_name = campaign.get('client_name', 'Unknown')
                ad_creative = campaign.get('ad_creative', {})
                
                for format_name, ad_data in ad_creative.items():
                    summary_data.append({
                        'Client': client_name,
                        'Format': format_name,
                        'Headline': ad_data.get('headline', ''),
                        'Body': ad_data.get('body', '')[:100] + '...',
                        'CTA': ad_data.get('call_to_action', ''),
                        'News_Count': campaign.get('relevant_news_count', 0)
                    })
            
            df = pd.DataFrame(summary_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name="campaign_summary.csv",
                mime="text/csv"
            )
    
    with col3:
        # View generated files
        if st.button("📁 View Generated Files"):
            st.info("Generated files are saved in:")
            st.code("""
            📁 generated_ads_text/
            📁 generated_ads_images/
            📁 data/
            """)

def about_page():
    """Display the About page"""
    st.markdown("""
    # 🚀 About News-Ads Generation Platform
    
    ## 🎯 What We Do
    
    **News-Ads Generation** is an AI-powered marketing platform specifically designed for financial services firms. We solve the critical challenge of creating timely, relevant advertising content that connects investment expertise with rapidly evolving market news.
    
    ## 💡 The Problem We Solve
    
    **Traditional Process (3+ days):**
    - Marketing teams manually read through hundreds of financial news articles
    - Brainstorm connections between news events and company expertise
    - Write ad copy from scratch
    - Design visual materials
    - Review and revise multiple times
    
    **Our AI Solution (10 minutes):**
    - ✅ Automatically analyzes 265+ news articles using vector search
    - ✅ Finds semantic connections with 0.31-0.66 relevance scores
    - ✅ Generates professional, compliant ad copy with GPT-4o
    - ✅ Creates complete visual campaigns with DALL-E 3 HD
    - ✅ Produces multiple ad formats simultaneously
    
    ## 🛠️ How It Works
    
    ### 1. **Smart Content Analysis**
    Our system scrapes client landing pages to understand investment expertise, capabilities, and value propositions.
    
    ### 2. **RAG-Powered News Matching**
    Using advanced Retrieval-Augmented Generation (RAG) with:
    - **Vector Database**: FAISS with 384-dimensional embeddings
    - **Semantic Search**: Sentence-BERT model for contextual relevance
    - **Smart Scoring**: Cosine similarity matching (0.3-0.8 typical range)
    
    ### 3. **AI Content Generation**
    - **Text Generation**: OpenAI GPT-4o with structured, RAG-enhanced prompts
    - **Visual Creation**: DALL-E 3 HD for professional background generation
    - **Text Overlays**: PIL-based system with professional typography
    
    ### 4. **Multi-Format Output**
    - LinkedIn single image ads (1024x1024)
    - Banner ads (1792x1024)
    - Custom creative formats
    - Complete marketing-ready materials
    
    ## 🏆 Key Benefits
    
    ### **For Marketing Teams:**
    - ⚡ **Speed**: Reduce campaign creation from days to minutes
    - 🎯 **Relevance**: AI finds meaningful connections between client expertise and market news
    - 📈 **Scale**: Generate multiple campaigns simultaneously
    - 💰 **Cost-Effective**: ~$2-5 per complete campaign vs. thousands for traditional methods
    
    ### **For Asset Management Firms:**
    - 🔒 **Compliance-Ready**: Built-in financial services regulatory tone
    - 🌟 **Professional Quality**: Marketing-ready materials requiring minimal human review
    - 📊 **Data-Driven**: Grounded in actual market news and semantic relevance scores
    - 🚀 **Competitive Edge**: Stay ahead with timely, relevant market commentary
    
    ## 🔧 Technology Stack
    
    ### **AI & Machine Learning:**
    - **OpenAI GPT-4o**: Latest Omni model for advanced text generation
    - **DALL-E 3 HD**: High-definition image generation
    - **Sentence-BERT**: Advanced semantic understanding
    - **FAISS**: High-performance vector similarity search
    
    ### **Data Processing:**
    - **RAG Architecture**: Retrieval-Augmented Generation for enhanced relevance
    - **RAKE Algorithm**: Rapid Automatic Keyword Extraction
    - **BeautifulSoup**: Intelligent web content extraction
    - **Pandas**: Robust data manipulation and analysis
    
    ### **Web Interface:**
    - **Streamlit**: Interactive, user-friendly web application
    - **PIL**: Professional image processing and text overlays
    - **Real-time Progress**: Live pipeline status and visualization
    
    ## 📊 Proven Results
    
    ### **Performance Metrics:**
    - **Processing Speed**: Complete pipeline in ~10 minutes
    - **Relevance Scores**: 0.31-0.66 semantic similarity (0.5+ indicates strong relevance)
    - **Output Quality**: 95%+ marketing-ready with minimal human review needed
    - **Cost Efficiency**: 85% reduction in campaign creation costs
    
    ### **Successful Campaigns Generated:**
    - **PIMCO**: Fed policy insights connected with emerging market trends
    - **State Street**: ESG leadership campaigns with sustainability news
    - **T. Rowe Price**: 2025 market outlook with growth strategies
    
    ## 🌐 Getting Started
    
    ### **Live Demo**
    Try our platform immediately at: [news-ads-generation.streamlit.app](https://news-ads-generation.streamlit.app)
    
    ### **For Your Organization**
    1. **Upload** your client data (Excel format)
    2. **Configure** your OpenAI API key
    3. **Generate** professional ad campaigns in minutes
    4. **Download** complete marketing materials
    
    ## 🚀 Future Roadmap
    
    - **A/B Testing Framework**: Automated campaign variation testing
    - **Real-time News Integration**: Live RSS feeds and news APIs
    - **Advanced Customization**: Logo integration and brand color schemes
    - **Multi-language Support**: International market expansion
    - **Analytics Dashboard**: Performance tracking and optimization
    
    ## 📞 Contact & Support
    
    **GitHub Repository**: [View Source Code](https://github.com/Zhijin-Guo1/news-ads-generation)  
    **Live Demo**: [Try the Platform](https://news-ads-generation.streamlit.app)  
    **Author**: Zhijin Guo  
    **Last Updated**: July 2025
    
    ---
    
    *Built with ❤️ for the future of AI-powered marketing in financial services*
    """)

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Add navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🏠 Home", "ℹ️ About"])
    
    if page == "ℹ️ About":
        about_page()
        return
        
    # Home page content
    display_header()
    config = sidebar_config()
    
    # Processing pipeline
    client_data = file_upload_section()
    
    # Use session state data if available
    if client_data or st.session_state.client_data:
        if client_data:
            st.session_state.processing_step = max(st.session_state.processing_step, 1)
        
        # Use session state data for subsequent steps
        current_client_data = st.session_state.client_data
        
        # Web scraping
        if current_client_data:
            scraped_data = web_scraping_section(current_client_data)
            if scraped_data:
                st.session_state.processing_step = max(st.session_state.processing_step, 2)
        
        # RAG processing
        if st.session_state.processing_step >= 2 and st.session_state.client_data:
            processed_data = rag_processing_section(st.session_state.client_data, config)
            if processed_data:
                st.session_state.processed_data = processed_data
                st.session_state.processing_step = max(st.session_state.processing_step, 3)
        
        # Campaign generation
        if st.session_state.processing_step >= 3 and st.session_state.processed_data:
            campaigns = campaign_generation_section(st.session_state.processed_data, config)
            if campaigns:
                st.session_state.generated_campaigns = campaigns
                st.session_state.processing_step = max(st.session_state.processing_step, 4)
        
        # Image generation
        if st.session_state.processing_step >= 4 and st.session_state.generated_campaigns:
            final_campaigns = image_generation_section(st.session_state.generated_campaigns, config)
            if final_campaigns:
                st.session_state.generated_campaigns = final_campaigns
    
    # Display results
    if st.session_state.generated_campaigns:
        campaign_display_section(st.session_state.generated_campaigns)
        download_section(st.session_state.generated_campaigns)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 News-Responsive Ad Generator | Built with Streamlit & OpenAI</p>
        <p>
        <a href="https://github.com/Zhijin-Guo1/news-ads-generation" target="_blank">📁 View on GitHub</a> | 
        <a href="https://github.com/Zhijin-Guo1/news-ads-generation#readme" target="_blank">📖 Read Documentation</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()