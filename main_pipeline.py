"""
Main Pipeline for Alphix ML Challenge: News-Responsive Ad Generation
Orchestrates the complete workflow from data parsing to ad generation
"""
import sys
import json
import time
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    import subprocess
    
    packages = [
        'pandas', 'openpyxl', 'requests', 'beautifulsoup4', 
        'sentence-transformers', 'rake-nltk', 'nltk', 
        'faiss-cpu', 'openai', 'scikit-learn', 'python-dotenv'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"  ⚠️  Failed to install {package}")
    
    # Download NLTK data
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    print("  ✅ Package installation complete")

def run_step(step_name: str, step_function, *args, **kwargs):
    """Run a pipeline step with error handling and timing"""
    print(f"\n🚀 {step_name}...")
    start_time = time.time()
    
    try:
        result = step_function(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"  ✅ {step_name} completed in {elapsed:.1f}s")
        return result, True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ {step_name} failed after {elapsed:.1f}s: {e}")
        return None, False

def check_input_files():
    """Check if required input files exist"""
    required_files = [
        'URL_and_news_articles_examples_by_client.xlsx',
        'Alphix_ML_Challenge_News_Ad_Generation.docx'
    ]
    
    print("📋 Checking input files...")
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"  ✅ Found {file}")
    
    if missing_files:
        print(f"  ❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("🎯 ALPHIX ML CHALLENGE: NEWS-RESPONSIVE AD GENERATION")
    print("🤖 RAG-Enabled Pipeline with OpenAI Integration")
    print("=" * 60)
    
    # Check prerequisites
    if not check_input_files():
        print("\n❌ Missing required input files. Please ensure you have:")
        print("   - URL_and_news_articles_examples_by_client.xlsx")
        print("   - Alphix_ML_Challenge_News_Ad_Generation.docx")
        return
    
    # Install requirements
    install_requirements()
    
    # Import modules after installation
    try:
        from utils.parse_client_data import parse_client_data
        from utils.web_scraper import scrape_text_from_url
        from utils.rag_processor import process_client_data_with_rag
        from openai_ad_generator import generate_complete_campaign
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed.")
        return
    
    pipeline_success = True
    
    # Step 1: Parse client data from Excel
    print("\n" + "="*50)
    parsed_data, success = run_step(
        "STEP 1: Parsing client data from Excel",
        lambda: parse_client_data('URL_and_news_articles_examples_by_client.xlsx')
    )
    
    if not success:
        return
    
    # Save parsed data
    with open('parsed_client_data.json', 'w') as f:
        json.dump(parsed_data, f, indent=2)
    
    print(f"  📊 Parsed data for {len(parsed_data)} clients")
    for client in parsed_data:
        print(f"    - {client['client_name']}: {len(client['news_articles'])} news articles")
    
    # Step 2: Scrape landing page content
    print("\n" + "="*50)
    print("🚀 STEP 2: Scraping client landing pages...")
    
    scraped_count = 0
    for client in parsed_data:
        url = client['url']
        print(f"  🌐 Scraping {client['client_name']}: {url}")
        
        content = scrape_text_from_url(url)
        client['landing_page_content'] = content
        
        if content:
            scraped_count += 1
            print(f"    ✅ Success: {len(content)} characters")
        else:
            print(f"    ⚠️  Failed to scrape content")
    
    # Save data with content
    with open('client_data_with_content.json', 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"  📊 Successfully scraped {scraped_count}/{len(parsed_data)} landing pages")
    
    # Step 3: Build RAG system and process data
    print("\n" + "="*50)
    rag_result, success = run_step(
        "STEP 3: Building RAG vector database and processing",
        process_client_data_with_rag,
        'client_data_with_content.json'
    )
    
    if not success:
        pipeline_success = False
    else:
        processed_clients = rag_result['processed_clients']
        print(f"  🔍 Processed {len(processed_clients)} clients with RAG")
        
        for client in processed_clients:
            rel_news = len(client.get('relevant_news', []))
            keywords = len(client.get('landing_page_keywords', []))
            print(f"    - {client['client_name']}: {rel_news} relevant news, {keywords} keywords")
    
    # Step 4: Generate ad campaigns
    print("\n" + "="*50)
    campaigns, success = run_step(
        "STEP 4: Generating AI-powered ad campaigns",
        generate_complete_campaign,
        'processed_client_data_rag.json'
    )
    
    if not success:
        pipeline_success = False
    else:
        print(f"  🎨 Generated campaigns for {len(campaigns)} clients")
        
        # Show sample results
        for campaign in campaigns[:2]:  # Show first 2 for brevity
            client_name = campaign.get('client_name', 'Unknown')
            ad_creative = campaign.get('ad_creative', {})
            
            print(f"\n  📢 Sample: {client_name}")
            if 'linkedin_single_image' in ad_creative:
                linkedin_ad = ad_creative['linkedin_single_image']
                print(f"    Headline: '{linkedin_ad.get('headline', '')[:50]}...'")
                print(f"    Body: '{linkedin_ad.get('body', '')[:60]}...'")
    
    # Final summary
    print("\n" + "="*60)
    if pipeline_success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated Files:")
        files = [
            ('parsed_client_data.json', 'Parsed Excel data'),
            ('client_data_with_content.json', 'Data with scraped content'),
            ('processed_client_data_rag.json', 'RAG-processed data'),
            ('generated_ad_campaigns.json', 'Final ad campaigns'),
            ('vector_index.faiss', 'Vector database index'),
            ('vector_metadata.pkl', 'Vector database metadata')
        ]
        
        for filename, description in files:
            if Path(filename).exists():
                print(f"  📁 {filename} - {description}")
        
        print(f"\n🎯 KEY RESULTS:")
        print(f"  • Processed {len(parsed_data)} clients")
        print(f"  • Scraped {scraped_count} landing pages")
        if campaigns:
            total_ads = sum(len(c.get('ad_creative', {})) for c in campaigns)
            print(f"  • Generated {total_ads} ad creative variations")
            print(f"  • Built vector database with semantic search")
        
        print(f"\n📖 NEXT STEPS:")
        print(f"  1. Review generated campaigns in 'generated_ad_campaigns.json'")
        print(f"  2. Replace 'xxxxx' in openai_ad_generator.py with real OpenAI API key")
        print(f"  3. Run individual components for customization")
        print(f"  4. Implement compliance checks and A/B testing")
        
    else:
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
        print("Some steps failed. Check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main()