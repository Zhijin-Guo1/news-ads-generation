import requests
from bs4 import BeautifulSoup

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
    # Example usage (will be replaced by actual client URLs)
    test_url = "https://www.ssga.com/uk/en_gb/institutional/capabilities/esg"
    scraped_content = scrape_text_from_url(test_url)
    if scraped_content:
        with open("/home/ubuntu/test_scraped_content.txt", "w", encoding="utf-8") as f:
            f.write(scraped_content)
        print(f"Scraped content from {test_url} saved to /home/ubuntu/test_scraped_content.txt")


