#!/usr/bin/env python3
"""
Setup script to pre-download NLTK data for the news-ads-generation project.
This ensures NLTK resources are available without runtime downloads.
"""

import nltk
import os
import sys

def setup_nltk_data():
    """Download and setup all required NLTK data"""
    print("🔧 Setting up NLTK data...")
    
    # Set NLTK data path to local directory
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of required NLTK resources
    required_resources = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    print(f"📁 NLTK data directory: {nltk_data_dir}")
    
    # Download each resource
    for resource in required_resources:
        try:
            print(f"⬇️  Downloading {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            print(f"✅ {resource} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {resource}: {e}")
    
    # Verify downloads
    print("\n🔍 Verifying downloads...")
    for resource in required_resources:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
            elif resource == 'punkt' or resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet', paths=[nltk_data_dir])
            elif resource == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger', paths=[nltk_data_dir])
            print(f"✅ {resource} verified")
        except LookupError:
            print(f"⚠️  {resource} not found after download")
    
    print(f"\n🎯 NLTK setup complete!")
    print(f"Data saved to: {nltk_data_dir}")
    return nltk_data_dir

if __name__ == "__main__":
    setup_nltk_data()