import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a single line
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

if __name__ == '__main__':
    # Load parsed client data
    try:
        with open('parsed_client_data.json', 'r') as f:
            client_data = json.load(f)
    except FileNotFoundError:
        print("Please run parse_client_data.py first to generate parsed_client_data.json")
        exit(1)
    
    # Scrape all client URLs
    print("=== SCRAPING CLIENT LANDING PAGES ===")
    for client in client_data:
        url = client['url']
        print(f"\nScraping {client['client_name']}: {url}")
        
        content = scrape_text_from_url(url)
        client['landing_page_content'] = content
        
        if content:
            print(f"  ✓ Successfully scraped {len(content)} characters")
            # Save a sample for review
            with open(f"{client['client_name'].replace(' ', '_')}_content.txt", "w", encoding="utf-8") as f:
                f.write(content[:1000] + "..." if len(content) > 1000 else content)
        else:
            print(f"  ✗ Failed to scrape content")
    
    # Save updated data
    with open('client_data_with_content.json', 'w', encoding='utf-8') as f:
        json.dump(client_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nClient data with scraped content saved to client_data_with_content.json")


