# 🚀 Streamlit Community Cloud Deployment Guide

## Quick Deployment Steps

### 1. Go to Streamlit Community Cloud
Visit: https://share.streamlit.io/

### 2. Sign In
- Click "Sign in with GitHub"
- Authorize Streamlit to access your repositories

### 3. Deploy New App
- Click "New app" button
- Select your GitHub repository: `Zhijin-Guo1/news-generation`
- Set main file path: `streamlit/streamlit_app.py`
- Keep branch as `main`

### 4. Configure Environment Variables
- Click "Advanced settings" 
- Add secrets in TOML format:
```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 5. Deploy
- Click "Deploy!"
- Wait 2-5 minutes for initial deployment

## 🔑 Getting Your OpenAI API Key

1. Visit https://platform.openai.com/
2. Create account or sign in
3. Go to API Keys section
4. Create new secret key
5. Copy the key (starts with `sk-`)

## 📱 Accessing Your App

After deployment, you'll get a URL like:
`https://your-app-name.streamlit.app`

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Check that all files are committed to GitHub
   - Verify requirements.txt includes all dependencies

2. **API Key Not Working**
   - Ensure secrets are properly formatted in TOML
   - Check that key starts with `sk-`

3. **App Won't Start**
   - Check logs in Streamlit Cloud dashboard
   - Verify Python version in runtime.txt

### Need Help?
- Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud
- Contact support through Streamlit Community forum

## 🎯 Your App Features

Once deployed, your app will have:
- ✅ News scraping and RAG processing
- ✅ AI-powered ad generation
- ✅ Image generation and display (600px resolution)
- ✅ Downloadable campaign results
- ✅ No manual API key input needed

## 🔄 Auto-Updates

Any changes pushed to your GitHub `main` branch will automatically redeploy your app!