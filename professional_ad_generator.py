"""
Professional Ad Generator with Text Overlay
Generates complete ad campaigns with images and text overlays
"""
import json
import os
from openai import OpenAI
import requests
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed")

class ProfessionalAdGenerator:
    # Model Configuration - Using the best available OpenAI models
    IMAGE_MODEL = "dall-e-3"  # Latest DALL-E 3 model
    IMAGE_QUALITY = "hd"      # HD quality for professional marketing materials
    
    def __init__(self, api_key: str = None):
        """Initialize the professional ad generator"""
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-openai-api-key-here':
            print("⚠️  No OpenAI API key found for image generation")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI DALL-E client initialized")
        
        # Create directory structure
        self.setup_directories()
    
    def setup_directories(self):
        """Create organized folder structure"""
        self.dirs = {
            'data': Path('data'),
            'text': Path('generated_ads_text'),
            'images': Path('generated_ads_images'),
            'final': Path('generated_ads_images/final_ads')
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print("📁 Created directory structure:")
        for name, path in self.dirs.items():
            print(f"   {name}: {path}")
    
    def move_data_files(self):
        """Move existing data files to data folder"""
        data_files = [
            'parsed_client_data.json',
            'client_data_with_content.json', 
            'processed_client_data_rag.json',
            'vector_index.faiss',
            'vector_metadata.pkl'
        ]
        
        for file in data_files:
            if os.path.exists(file):
                try:
                    os.rename(file, f"data/{file}")
                    print(f"📦 Moved {file} → data/")
                except:
                    pass  # File might already be moved
    
    def enhance_image_prompt(self, description: str, client_name: str, ad_format: str) -> str:
        """Enhanced prompt for professional financial marketing images"""
        
        # Base professional styling
        base_prompt = f"Professional financial marketing image for {client_name}. "
        
        # Format-specific optimizations
        if "linkedin" in ad_format.lower():
            format_style = "LinkedIn social media style, square format, professional business aesthetic. "
        elif "banner" in ad_format.lower():
            format_style = "Web banner advertisement style, horizontal layout, clean design with space for text. "
        else:
            format_style = "Professional marketing material, clean corporate design. "
        
        # Enhanced description with professional keywords
        enhanced = f"{base_prompt}{format_style}{description}. "
        enhanced += "High-quality professional photography or clean vector graphics, "
        enhanced += "modern financial services aesthetic, sophisticated color palette with blues and greens, "
        enhanced += "corporate professional style, minimal and clean design, "
        enhanced += "no text or typography in the image, suitable for adding text overlay later, "
        enhanced += "institutional investor appropriate, premium brand feeling"
        
        return enhanced
    
    def generate_background_image(self, prompt: str, size: str = "1024x1024") -> str:
        """Generate background image using DALL-E (without text)"""
        if not self.client:
            print("❌ No OpenAI client available")
            return None
        
        try:
            print(f"🎨 Generating background image...")
            print(f"   Prompt: {prompt[:80]}...")
            
            response = self.client.images.generate(
                model=self.IMAGE_MODEL,
                prompt=prompt,
                size=size,
                quality=self.IMAGE_QUALITY,
                n=1,
            )
            
            image_url = response.data[0].url
            print("✅ Background image generated successfully")
            return image_url
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None
    
    def download_image(self, image_url: str, filename: str) -> bool:
        """Download image from URL"""
        try:
            response = requests.get(image_url, timeout=60)  # Longer timeout
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Background image saved: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return False
    
    def add_text_overlay(self, background_path: str, ad_data: dict, client_name: str, 
                        ad_format: str, output_path: str) -> bool:
        """Add professional text overlay to background image"""
        try:
            # Load background image
            img = Image.open(background_path)
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            width, height = img.size
            
            # Define colors and styling
            primary_color = (255, 255, 255)  # White text
            accent_color = (0, 120, 200)     # Blue accent
            shadow_color = (0, 0, 0, 128)    # Semi-transparent black
            
            # Try to load professional fonts, fallback to default
            try:
                if "linkedin" in ad_format.lower():
                    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
                    body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
                    cta_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
                elif "banner" in ad_format.lower():
                    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
                    body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                    cta_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                else:
                    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
                    body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
                    cta_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except:
                # Fallback to default font
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                cta_font = ImageFont.load_default()
            
            # Extract text content
            headline = ad_data.get('headline', '')
            body = ad_data.get('body', '')
            cta = ad_data.get('call_to_action', '')
            
            # Add semi-transparent overlay for text readability
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            if "linkedin" in ad_format.lower():
                # LinkedIn format - bottom text overlay
                overlay_height = height // 3
                overlay_draw.rectangle([(0, height - overlay_height), (width, height)], 
                                     fill=(0, 0, 0, 150))
                
                # Position text
                text_y = height - overlay_height + 20
                margin = 30
                
                # Headline
                wrapped_headline = textwrap.fill(headline, width=35)
                draw.text((margin, text_y), wrapped_headline, font=title_font, fill=primary_color)
                text_y += 80
                
                # Body text
                wrapped_body = textwrap.fill(body, width=45)
                draw.text((margin, text_y), wrapped_body, font=body_font, fill=primary_color)
                text_y += 100
                
                # CTA Button
                cta_width = len(cta) * 12 + 40
                cta_height = 40
                cta_x = margin
                cta_y = text_y
                
                # Draw button background
                draw.rectangle([(cta_x, cta_y), (cta_x + cta_width, cta_y + cta_height)], 
                             fill=accent_color)
                draw.text((cta_x + 20, cta_y + 10), cta, font=cta_font, fill=primary_color)
                
            elif "banner" in ad_format.lower():
                # Banner format - side text layout
                text_width = width // 2
                overlay_draw.rectangle([(width - text_width, 0), (width, height)], 
                                     fill=(0, 0, 0, 150))
                
                # Position text on right side
                text_x = width - text_width + 20
                text_y = 50
                
                # Headline
                wrapped_headline = textwrap.fill(headline, width=20)
                draw.text((text_x, text_y), wrapped_headline, font=title_font, fill=primary_color)
                text_y += 100
                
                # Body text
                wrapped_body = textwrap.fill(body, width=25)
                draw.text((text_x, text_y), wrapped_body, font=body_font, fill=primary_color)
                text_y += 80
                
                # CTA Button
                cta_width = len(cta) * 10 + 30
                cta_height = 35
                draw.rectangle([(text_x, text_y), (text_x + cta_width, text_y + cta_height)], 
                             fill=accent_color)
                draw.text((text_x + 15, text_y + 8), cta, font=cta_font, fill=primary_color)
            
            # Composite overlay onto main image
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
            
            # Add client logo/branding area (placeholder)
            logo_y = 20
            logo_x = width - 200
            draw.rectangle([(logo_x, logo_y), (logo_x + 180, logo_y + 40)], 
                         fill=(255, 255, 255, 200), outline=accent_color)
            draw.text((logo_x + 10, logo_y + 10), client_name, font=body_font, fill=accent_color)
            
            # Save final image
            img.convert('RGB').save(output_path, 'PNG', quality=95)
            print(f"🎨 Final ad created: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding text overlay: {e}")
            return False
    
    def generate_complete_ad_campaign(self, campaigns_file: str = 'generated_ad_campaigns.json'):
        """Generate complete ad campaign with images and text"""
        
        # Move data files to organized structure
        self.move_data_files()
        
        # Load campaigns
        try:
            with open(campaigns_file, 'r') as f:
                campaigns = json.load(f)
        except FileNotFoundError:
            print(f"❌ Campaign file not found: {campaigns_file}")
            return
        
        # Save text campaigns to organized folder
        text_output = self.dirs['text'] / 'ad_campaigns.json'
        with open(text_output, 'w') as f:
            json.dump(campaigns, f, indent=2)
        print(f"📝 Text campaigns saved: {text_output}")
        
        generated_ads = []
        
        for campaign in campaigns:
            client_name = campaign.get('client_name', 'Client')
            ad_creative = campaign.get('ad_creative', {})
            
            print(f"\n🎯 Creating complete ads for {client_name}")
            print("=" * 50)
            
            for ad_format, ad_data in ad_creative.items():
                if not isinstance(ad_data, dict):
                    continue
                
                image_description = ad_data.get('image_description', '')
                if not image_description:
                    continue
                
                print(f"\n📢 Processing {ad_format}")
                
                # Enhanced prompt for background image (no text)
                enhanced_prompt = self.enhance_image_prompt(
                    image_description, client_name, ad_format
                )
                
                # Determine size based on format
                if "banner" in ad_format.lower():
                    size = "1792x1024"  # Wide banner
                elif "linkedin" in ad_format.lower():
                    size = "1024x1024"  # Square
                else:
                    size = "1024x1024"
                
                # Generate background image
                image_url = self.generate_background_image(enhanced_prompt, size)
                
                if image_url:
                    # Create filenames
                    safe_client = client_name.replace(' ', '_').replace('.', '')
                    safe_format = ad_format.replace(' ', '_')
                    timestamp = datetime.now().strftime("%H%M%S")
                    
                    # Background image
                    bg_filename = self.dirs['images'] / f"{safe_client}_{safe_format}_bg_{timestamp}.png"
                    
                    # Final ad with text
                    final_filename = self.dirs['final'] / f"{safe_client}_{safe_format}_final_{timestamp}.png"
                    
                    # Download background
                    if self.download_image(image_url, str(bg_filename)):
                        
                        # Add text overlay
                        if self.add_text_overlay(str(bg_filename), ad_data, client_name, 
                                               ad_format, str(final_filename)):
                            
                            generated_ads.append({
                                'client': client_name,
                                'ad_format': ad_format,
                                'background_image': str(bg_filename),
                                'final_ad': str(final_filename),
                                'headline': ad_data.get('headline', ''),
                                'body': ad_data.get('body', ''),
                                'cta': ad_data.get('call_to_action', ''),
                                'size': size,
                                'generated_at': datetime.now().isoformat()
                            })
                
                # Rate limiting for API calls
                print("⏳ Waiting 3 seconds for next generation...")
                time.sleep(3)
        
        # Save metadata
        metadata_file = self.dirs['final'] / 'ad_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(generated_ads, f, indent=2)
        
        print(f"\n🎉 CAMPAIGN GENERATION COMPLETE!")
        print(f"✅ Generated {len(generated_ads)} complete ads")
        print(f"📁 Final ads: {self.dirs['final']}/")
        print(f"📄 Metadata: {metadata_file}")
        
        return generated_ads

def main():
    """Main function"""
    print("🎨 PROFESSIONAL AD GENERATOR")
    print("Generates complete marketing campaigns with images and text overlays")
    print("=" * 70)
    
    generator = ProfessionalAdGenerator()
    
    if not generator.client:
        print("\n❌ Cannot generate images without OpenAI API key")
        print("Please ensure your API key is set in .env file")
        return
    
    try:
        ads = generator.generate_complete_ad_campaign()
        
        if ads:
            print(f"\n📊 GENERATION SUMMARY:")
            print(f"🎯 Total ads created: {len(ads)}")
            
            for ad in ads:
                print(f"\n📢 {ad['client']} - {ad['ad_format']}")
                print(f"   📝 Headline: {ad['headline'][:50]}...")
                print(f"   🎨 Final ad: {ad['final_ad']}")
                print(f"   📏 Size: {ad['size']}")
        
        print(f"\n🚀 Ready for marketing deployment!")
        
    except Exception as e:
        print(f"❌ Error in campaign generation: {e}")

if __name__ == "__main__":
    main()