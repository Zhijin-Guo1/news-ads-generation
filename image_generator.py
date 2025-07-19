"""
AI Image Generator for Ad Campaigns
Uses OpenAI DALL-E to generate professional marketing images
"""
import json
import os
from openai import OpenAI
import requests
from datetime import datetime
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed")

class ImageGenerator:
    def __init__(self, api_key: str = None):
        """Initialize DALL-E image generator"""
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-openai-api-key-here':
            print("⚠️  No OpenAI API key found for image generation")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI DALL-E client initialized")
    
    def enhance_image_prompt(self, description: str, client_name: str, ad_format: str) -> str:
        """Enhance image description for better DALL-E results"""
        
        # Base enhancement for professional financial marketing
        base_prompt = f"Professional financial marketing image for {client_name}. "
        
        # Add format-specific styling
        if "linkedin" in ad_format.lower():
            format_style = "LinkedIn social media style, clean and professional, suitable for business networking. "
        elif "banner" in ad_format.lower():
            format_style = "Web banner advertisement style, clean layout with space for text overlay. "
        else:
            format_style = "Professional marketing material style. "
        
        # Enhance with professional financial imagery keywords
        enhanced = f"{base_prompt}{format_style}{description}. "
        enhanced += "High quality, professional photography or clean vector graphics, "
        enhanced += "modern financial services aesthetic, blue and green color palette, "
        enhanced += "corporate professional style, no text overlay, "
        enhanced += "suitable for institutional investors"
        
        return enhanced
    
    def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        """Generate image using DALL-E"""
        if not self.client:
            print("❌ No OpenAI client available for image generation")
            return None
        
        try:
            print(f"🎨 Generating image with prompt: {prompt[:100]}...")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            print("✅ Image generated successfully")
            return image_url
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None
    
    def download_image(self, image_url: str, filename: str) -> bool:
        """Download image from URL to local file"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Image saved: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return False
    
    def generate_campaign_images(self, campaigns_file: str = 'generated_ad_campaigns.json'):
        """Generate images for all ad campaigns"""
        
        # Create images directory
        images_dir = "generated_images"
        os.makedirs(images_dir, exist_ok=True)
        print(f"📁 Created directory: {images_dir}")
        
        # Load campaigns
        try:
            with open(campaigns_file, 'r') as f:
                campaigns = json.load(f)
        except FileNotFoundError:
            print(f"❌ Campaign file not found: {campaigns_file}")
            return
        
        generated_images = []
        
        for campaign in campaigns:
            client_name = campaign.get('client_name', 'Client')
            ad_creative = campaign.get('ad_creative', {})
            
            print(f"\n🎯 Generating images for {client_name}")
            print("-" * 40)
            
            for ad_format, ad_data in ad_creative.items():
                if not isinstance(ad_data, dict):
                    continue
                
                image_description = ad_data.get('image_description', '')
                if not image_description:
                    continue
                
                # Enhance prompt for better results
                enhanced_prompt = self.enhance_image_prompt(
                    image_description, client_name, ad_format
                )
                
                # Determine image size based on format
                if "banner" in ad_format.lower():
                    size = "1792x1024"  # Wide format for banners
                elif "linkedin" in ad_format.lower():
                    size = "1024x1024"  # Square format for LinkedIn
                else:
                    size = "1024x1024"  # Default square
                
                # Generate image
                image_url = self.generate_image(enhanced_prompt, size)
                
                if image_url:
                    # Create filename
                    safe_client = client_name.replace(' ', '_').replace('.', '')
                    safe_format = ad_format.replace(' ', '_')
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"{images_dir}/{safe_client}_{safe_format}_{timestamp}.png"
                    
                    # Download image
                    if self.download_image(image_url, filename):
                        generated_images.append({
                            'client': client_name,
                            'ad_format': ad_format,
                            'filename': filename,
                            'original_description': image_description,
                            'enhanced_prompt': enhanced_prompt,
                            'headline': ad_data.get('headline', ''),
                            'size': size
                        })
                
                # Rate limiting - be respectful with API
                time.sleep(2)
        
        # Save metadata
        metadata_file = f"{images_dir}/image_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(generated_images, f, indent=2)
        
        print(f"\n✅ Generated {len(generated_images)} images")
        print(f"📁 Images saved in: {images_dir}/")
        print(f"📄 Metadata saved: {metadata_file}")
        
        return generated_images

def main():
    """Main function to generate all campaign images"""
    print("🎨 AI IMAGE GENERATOR FOR AD CAMPAIGNS")
    print("=" * 50)
    
    # Initialize generator
    generator = ImageGenerator()
    
    if not generator.client:
        print("\n❌ Cannot generate images without OpenAI API key")
        print("Please ensure your API key is set in .env file")
        return
    
    # Generate images
    try:
        images = generator.generate_campaign_images()
        
        if images:
            print(f"\n🎉 SUCCESS! Generated {len(images)} professional marketing images")
            print("\n📊 Generated Images Summary:")
            
            for img in images:
                print(f"  📷 {img['client']} - {img['ad_format']}")
                print(f"     File: {img['filename']}")
                print(f"     Headline: {img['headline'][:50]}...")
                print(f"     Size: {img['size']}")
                print()
        else:
            print("⚠️  No images were generated")
            
    except Exception as e:
        print(f"❌ Error in image generation: {e}")

if __name__ == "__main__":
    main()