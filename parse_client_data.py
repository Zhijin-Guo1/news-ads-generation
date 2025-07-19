import pandas as pd
import json
from typing import Dict, List, Any

def parse_client_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Excel file containing client URLs and related news articles
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of client data dictionaries
    """
    # Read all sheets from Excel file
    excel_data = pd.read_excel(file_path, sheet_name=None)
    
    client_data = []
    
    for sheet_name, df in excel_data.items():
        print(f"Processing sheet: {sheet_name}")
        
        # Find the URL row
        url = None
        news_articles = []
        
        for idx, row in df.iterrows():
            # Look for URL in first column
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).startswith('URL -'):
                url = str(row.iloc[0]).replace('URL -', '').strip()
                continue
            
            # Look for news article headers (Title, Source, Published date, URL)
            if (pd.notna(row.iloc[0]) and 
                str(row.iloc[0]).strip().lower() == 'title' and
                len([x for x in row if pd.notna(x)]) >= 3):
                continue  # Skip header row
            
            # Parse news articles
            if (pd.notna(row.iloc[0]) and 
                str(row.iloc[0]).strip() and 
                str(row.iloc[0]).strip().lower() not in ['title', 'total', 'nan']):
                
                title = str(row.iloc[0]).strip()
                source = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
                pub_date = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
                article_url = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
                
                if title and title != 'nan':
                    news_articles.append({
                        "title": title,
                        "source": source,
                        "published_date": pub_date,
                        "url": article_url
                    })
        
        if url and news_articles:
            client_data.append({
                "client_name": sheet_name,
                "url": url,
                "news_articles": news_articles
            })
            print(f"Found {len(news_articles)} news articles for {sheet_name}")
    
    return client_data

if __name__ == '__main__':
    parsed_data = parse_client_data('URL_and_news_articles_examples_by_client.xlsx')
    with open('parsed_client_data.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)
    print("Parsed data saved to parsed_client_data.json")
    
    # Print summary
    print(f"\n=== PARSED DATA SUMMARY ===")
    for client in parsed_data:
        print(f"\nClient: {client['client_name']}")
        print(f"URL: {client['url']}")
        print(f"News articles: {len(client['news_articles'])}")
        
        # Show first few news titles
        for i, article in enumerate(client['news_articles'][:3]):
            print(f"  {i+1}. {article['title'][:80]}...")


