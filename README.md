# Alphix ML Engineering Challenge: News-Responsive Ad Generation

This repository contains a technical approach and a simulated prototype for an LLM-powered tool designed to generate context-aware ad creative for asset management firms. The solution aims to dynamically link a client's message (derived from their landing page) with the current news cycle, producing timely and resonant marketing assets across various digital ad formats.

## Project Overview

The core problem addressed is the manual, time-consuming, and often reactive process of creating marketing content that is relevant to both the client's core message and the rapidly evolving financial news landscape. This solution automates a significant portion of the creative process, focusing on timeliness, relevance, efficiency, compliance, and scalability.

## Solution Architecture

The proposed architecture is modular, integrating components for data ingestion, processing, LLM interaction, and output generation. Key components include:

*   **Data Ingestion Layer:** Fetches content from client landing page URLs and processes news headlines/summaries from an Excel file.
*   **Meaning Extraction and Semantic Alignment Layer:** Utilizes embedding models (Sentence-BERT) and keyword extraction (RAKE) to understand the semantic meaning of both client landing pages and news articles, and to identify relevant news items.
*   **LLM Orchestration Layer (Simulated):** In a real-world scenario, this layer would interact with a Large Language Model (LLM) like GPT-4 or Claude. For this prototype, the LLM interaction is simulated to demonstrate the expected output structure and content generation logic.
*   **Image Generation Service (Simulated):** Similarly, image generation is simulated, providing descriptive suggestions for ad imagery.
*   **Ad Format Templating and Visualization Layer:** Assembles the generated ad copy and imagery suggestions into various digital ad formats (e.g., LinkedIn single image, banner ads).

## How it Works

1.  **Data Ingestion:**
    *   The `Alphix_ML_Challenge_News_Ad_Generation.docx` file provides the challenge description.
    *   The `MLEngineer-URLandnewsarticlesexamplesbyclient.xlsx` file contains example client URLs and related news articles.
    *   The `parse_client_data.py` script reads the Excel file and extracts client URLs and associated news articles into a structured JSON format.

2.  **Web Scraping:**
    *   The `web_scraper.py` script is used to fetch and extract text content from the provided client landing page URLs. For demonstration, it scrapes a single example URL.

3.  **NLP Processing:**
    *   The `nlp_processor.py` script takes the scraped landing page content and news articles.
    *   It uses `SentenceTransformer` to generate embeddings for both the landing page text and news article titles/sources, enabling semantic similarity calculations.
    *   It employs `RAKE` (Rapid Automatic Keyword Extraction) to extract key phrases from both sources.
    *   The processed data, including embeddings and keywords, is saved to `processed_client_data.json`.

4.  **Ad Generation (Simulated LLM):**
    *   The `ad_generator.py` script takes the processed client data.
    *   It calculates the cosine similarity between the landing page embedding and each news article embedding to identify the most relevant news.
    *   It then simulates the LLM's role by generating ad copy (headline, body, call-to-action) and imagery suggestions for specified ad formats (LinkedIn single image, banner ad). The ad copy incorporates keywords from the landing page and relevant news titles.
    *   The generated ad creative is saved to `generated_ads.json`.

## Files in this Repository

*   `Alphix_ML_Challenge_News_Ad_Generation.docx`: Original challenge document.
*   `MLEngineer-URLandnewsarticlesexamplesbyclient.xlsx`: Original data file with client URLs and news examples.
*   `solution_design.md`: Detailed technical design document outlining the approach, architecture, and methodology.
*   `parse_client_data.py`: Python script to parse the Excel data.
*   `web_scraper.py`: Python script for web scraping landing page content.
*   `nlp_processor.py`: Python script for NLP tasks (embedding generation, keyword extraction).
*   `ad_generator.py`: Python script simulating LLM-based ad generation.
*   `parsed_client_data.json`: Output of `parse_client_data.py`.
*   `test_scraped_content.txt`: Example scraped content from a landing page.
*   `processed_client_data.json`: Output of `nlp_processor.py`.
*   `generated_ads.json`: Output of `ad_generator.py` (simulated ad creative).

## Setup and Running the Prototype

To run the scripts, you will need Python 3.x and the following libraries:

```bash
pip install pandas openpyxl requests beautifulsoup4 sentence-transformers rake_nltk nltk
python3 -c "import nltk; nltk.download(\"stopwords\"); nltk.download(\"punkt\"); nltk.download(\"punkt_tab\")"
```

Then, execute the scripts in the following order:

```bash
python3 parse_client_data.py
python3 web_scraper.py
python3 nlp_processor.py
python3 ad_generator.py
```

The final generated ads will be in `generated_ads.json`.

## Future Enhancements

*   **Integration with Real LLMs:** Replace simulated LLM calls with actual API calls to models like GPT-4 or Claude.
*   **Advanced Keyword/Theme Extraction:** Implement more sophisticated NLP techniques for deeper understanding of content.
*   **Image Generation Integration:** Connect with DALL·E or Midjourney APIs for actual image generation.
*   **Ad Format Templates:** Develop robust templating for various digital ad platforms.
*   **Evaluation Metrics:** Implement automated and human-in-the-loop evaluation frameworks for continuous improvement.
*   **User Interface:** Build a simple web interface for inputting client data and viewing generated ads.

---

**Author:** Manus AI
**Date:** 2025-07-19


