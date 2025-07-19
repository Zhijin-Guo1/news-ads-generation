import csv
import json

def parse_client_data(file_path):
    client_data = []
    current_url_entry = None
    news_article_headers = ["Title", "Source", "Published date", "URL"]

    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(reader):
            # Skip empty rows
            if not any(row):
                continue

            # Check for URL line
            if row[0].startswith("URL - "):
                if current_url_entry:
                    client_data.append(current_url_entry)
                current_url = row[0].replace("URL - ", "").strip()
                # The URL might have trailing commas in the original Excel, so take only the first part
                current_url = current_url.split(",")[0].strip()
                current_url_entry = {"url": current_url, "news_articles": []}
            # Check for news article header
            elif row == news_article_headers:
                continue # Skip the header row
            # Check for news article data
            elif current_url_entry and len(row) >= 4 and row[0].lower() != "total":
                # Ensure that the row is not the \'Total\' row
                title = row[0].strip()
                source = row[1].strip()
                published_date = row[2].strip()
                url = row[3].strip()
                current_url_entry["news_articles"].append({"title": title, "source": source, "published_date": published_date, "url": url})
            elif row and row[0].strip().lower() == "total":
                # End of news articles for the current URL
                continue

    if current_url_entry:
        client_data.append(current_url_entry)

    return client_data

if __name__ == '__main__':
    parsed_data = parse_client_data('/home/ubuntu/upload/client_data.csv')
    with open('/home/ubuntu/parsed_client_data.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)
    print("Parsed data saved to /home/ubuntu/parsed_client_data.json")


