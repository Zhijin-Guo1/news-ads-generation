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
    st.markdown('<h1 class="main-header">🎯 News-Responsive Ad Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🤖 AI-Powered Marketing Campaign Generation</h3>
        <p>Transform client data and news articles into professional, contextually relevant ad campaigns using 
        RAG (Retrieval-Augmented Generation) and OpenAI's latest models.</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_config():
    """Configure the sidebar with settings and information"""
    st.sidebar.header("⚙️ Configuration")
    
    # Use API key from Streamlit secrets or environment variable
    api_key = ""
    
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.sidebar.success("✅ API Key configured from Streamlit secrets")
        else:
            # Fallback to environment variable
            api_key = os.getenv('OPENAI_API_KEY', '')
            if api_key:
                st.sidebar.success("✅ API Key configured from environment")
            else:
                raise KeyError("No API key found")
    except (KeyError, AttributeError, Exception) as e:
        st.sidebar.error("❌ OPENAI_API_KEY not found")
        st.sidebar.info("💡 For Streamlit Cloud: Add OPENAI_API_KEY in app secrets\n💡 For local: Set OPENAI_API_KEY environment variable")
        
        # Debug information
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
            st.write(f"- Error details: {str(e)}")
    
    st.session_state.api_key = api_key
    
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
        help="Number of most relevant news articles to use"
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
    
    if st.button("🔍 Build Vector Database", type="primary"):
        with st.spinner("🧠 Building vector database and processing with RAG..."):
            try:
                # Initialize RAG processor
                rag_processor = RAGProcessor()
                
                # Build vector database
                rag_processor.build_vector_database(client_data)
                
                # Process each client
                processed_clients = []
                
                for client in client_data:
                    if client.get('landing_page_content'):
                        # Find relevant news
                        relevant_news = rag_processor.find_relevant_news(
                            client['client_name'],
                            client['landing_page_content'],
                            k=config['max_news_articles']
                        )
                        
                        # Extract keywords
                        keywords = rag_processor.extract_keywords(
                            client['landing_page_content'], 
                            max_keywords=10
                        )
                        
                        client['relevant_news'] = relevant_news
                        client['landing_page_keywords'] = keywords
                        
                        processed_clients.append(client)
                
                st.success(f"✅ RAG processing completed for {len(processed_clients)} clients")
                
                # Display RAG results
                for client in processed_clients:
                    with st.expander(f"📊 RAG Results: {client['client_name']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔑 Keywords")
                            for keyword in client['landing_page_keywords'][:5]:
                                st.write(f"• {keyword}")
                        
                        with col2:
                            st.subheader("📰 Relevant News")
                            for news in client['relevant_news'][:3]:
                                score = news.get('similarity_score', 0)
                                st.write(f"• **{news['title'][:50]}...** (Score: {score:.3f})")
                
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
                
                # Save campaigns
                os.makedirs("generated_ads_text", exist_ok=True)
                with open("generated_ads_text/ad_campaigns.json", 'w') as f:
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
                    return generated_ads
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

def main():
    """Main Streamlit application"""
    init_session_state()
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