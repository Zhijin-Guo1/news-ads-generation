# Deployment Guide for News-Responsive Ad Generator

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

**Free and Easy Deployment:**

1. **Fork/Clone Repository** to your GitHub account
2. **Visit** [share.streamlit.io](https://share.streamlit.io)
3. **Connect GitHub** and select this repository
4. **Set Environment Variables:**
   - Add `OPENAI_API_KEY` in the Streamlit Cloud secrets
5. **Deploy** - Your app will be live at `https://your-app-name.streamlit.app`

**Streamlit Cloud Configuration:**
```toml
# In Streamlit Cloud secrets
OPENAI_API_KEY = "sk-your-openai-key-here"
```

### Option 2: Heroku Deployment

**Requirements:**
- Heroku account
- Heroku CLI installed

**Steps:**
```bash
# 1. Create Heroku app
heroku create your-app-name

# 2. Set environment variables
heroku config:set OPENAI_API_KEY=sk-your-key-here

# 3. Create Procfile
echo "web: streamlit run streamlit_app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile

# 4. Deploy
git push heroku main
```

### Option 3: Local Development

**For Development and Testing:**
```bash
# Clone repository
git clone https://github.com/Zhijin-Guo1/news-generation.git
cd news-generation

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export OPENAI_API_KEY=sk-your-key-here

# Run locally
streamlit run streamlit_app.py
```

**Access at:** `http://localhost:8501`

### Option 4: Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

**Deploy:**
```bash
# Build image
docker build -t news-ad-generator .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key news-ad-generator
```

## 🔐 Security Considerations

### Environment Variables
- **Never commit** API keys to the repository
- **Use environment variables** for all sensitive data
- **Rotate keys** regularly for production use

### Production Setup
```bash
# Use secrets management
export OPENAI_API_KEY=sk-your-production-key
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 📊 Performance Optimization

### For Production Deployment:
1. **Resource Limits**: Set appropriate memory/CPU limits
2. **Caching**: Streamlit handles caching automatically
3. **Rate Limiting**: OpenAI API has rate limits
4. **File Storage**: Consider cloud storage for large files

### Recommended Instance Specs:
- **Memory**: 2GB minimum (for vector database)
- **CPU**: 1-2 cores
- **Storage**: 5GB (for generated images)

## 🔧 Troubleshooting

### Common Issues:

**1. Import Errors:**
```bash
pip install -r requirements.txt
```

**2. API Key Issues:**
```bash
# Check environment variable
echo $OPENAI_API_KEY
```

**3. Memory Issues:**
```bash
# Reduce vector database size
# Limit image generation
```

**4. Port Conflicts:**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

## 🚀 Quick Deploy Links

### Streamlit Cloud
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### Heroku
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

---

**Note**: For production deployment, ensure you have appropriate API quotas and consider implementing user authentication if needed.