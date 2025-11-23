# 🚀 Render Deployment Guide - RAGVoice AI Agent

This guide walks you through deploying your LiveKit multi-agent voice assistant to Render's free tier.

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Render account (sign up at https://render.com)
- ✅ LiveKit Cloud account (for room management)
- ✅ API keys: OpenAI, Deepgram, Cartesia, WeatherAPI

---

## 🔧 Step-by-Step Deployment

### **Step 1: Prepare Your Repository**

1. **Commit the updated `render.yaml`**:
   ```bash
   git add render.yaml DEPLOYMENT_GUIDE.md
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Verify `.gitignore` protects secrets**:
   - Ensure `.env.local` is NOT pushed to GitHub
   - Your `.gitignore` already has this configured ✅

---

### **Step 2: Create Environment Group in Render**

1. **Login to Render Dashboard**: https://dashboard.render.com

2. **Navigate to Environment Groups**:
   - Click **"Environment"** in left sidebar
   - Click **"New Environment Group"**

3. **Create group named**: `ragvoice-ai-env`

4. **Add all required environment variables**:

   ```bash
   # LiveKit Configuration (REQUIRED)
   LIVEKIT_URL=your_livekit_url_here
   LIVEKIT_API_KEY=your_livekit_api_key_here
   LIVEKIT_API_SECRET=your_livekit_api_secret_here

   # AI Service API Keys (REQUIRED)
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   CARTESIA_API_KEY=your_cartesia_api_key_here

   # Tool API Keys (OPTIONAL - for weather/currency tools)
   WEATHER_API_KEY=your_weather_api_key_here
   ```

   ⚠️ **Security Note**: Use your actual keys from `.env.local` file!

5. **Click "Save"**

---

### **Step 3: Deploy to Render**

#### **Option A: Deploy via Blueprint (Recommended)**

1. **Go to Render Dashboard**: https://dashboard.render.com

2. **Click "New +"** → **"Blueprint"**

3. **Connect Your Repository**:
   - Click **"Connect GitHub"** (if not connected)
   - Authorize Render to access your repositories
   - Select: `george07-t/RAGVoice-AI`

4. **Configure Blueprint**:
   - Render will automatically detect `render.yaml`
   - Service name: `ragvoice-ai-agent`
   - Plan: **Free** (or Starter for better performance)
   - Environment Group: `ragvoice-ai-env` ✅

5. **Click "Apply"** to start deployment

#### **Option B: Manual Deployment**

1. **Go to Render Dashboard** → **"New +"** → **"Private Service"**

2. **Connect Repository**: Select `george07-t/RAGVoice-AI`

3. **Configure Service**:
   - **Name**: `ragvoice-ai-agent`
   - **Region**: Virginia (US East)
   - **Branch**: `main`
   - **Runtime**: Docker
   - **Dockerfile Path**: `Dockerfile` (default)
   - **Docker Build Context**: `.` (root directory)

4. **Advanced Settings**:
   - **Environment Group**: Select `ragvoice-ai-env`
   - **Health Check Path**: `/health` (optional)
   - **Auto-Deploy**: ✅ Enabled (deploys on git push)

5. **Plan Selection**:
   - **Free Plan**: 512 MB RAM, 0.1 CPU (may have cold starts)
   - **Starter Plan**: $7/month, 512 MB RAM, 0.5 CPU (recommended)

6. **Click "Create Private Service"**

---

### **Step 4: Monitor Deployment**

1. **Watch Build Logs**:
   - Render will build your Docker image
   - This takes ~5-10 minutes first time
   - Logs show: Dependencies installation, model downloads, etc.

2. **Deployment Status**:
   - 🟢 **Live**: Service running successfully
   - 🟡 **Building**: Docker image building
   - 🔴 **Failed**: Check logs for errors

3. **Check Agent Logs**:
   ```
   Starting pre-warm sequence...
   ✓ Metrics logger initialized
   Loading VAD model...
   Pre-warming RAG system...
   ✅ RAG system pre-warmed successfully
   ```

---

### **Step 5: Connect to LiveKit Room**

Your agent is now deployed! To test:

1. **Create LiveKit Room** (via LiveKit Cloud dashboard):
   - Go to: https://cloud.livekit.io
   - Create a new room: `test-room`

2. **Your Render agent will automatically**:
   - Connect to the room when a user joins
   - Process voice input via Deepgram
   - Route queries through multi-agent system
   - Generate voice responses via Cartesia

3. **Test with LiveKit Playground**:
   - https://meet.livekit.io
   - Enter your room details
   - Start speaking to test the agent

---

## 🔍 Troubleshooting

### **Issue: Deployment Failed**

**Check Logs**:
```bash
# In Render dashboard, go to Logs tab
# Look for errors like:
- Missing environment variables
- Docker build failures
- Python package installation errors
```

**Common Fixes**:
- Verify all environment variables are set in `ragvoice-ai-env` group
- Ensure GitHub repository is public or Render has access
- Check Dockerfile syntax (already validated ✅)

---

### **Issue: Agent Not Responding**

**Verify**:
1. LiveKit URL/API Key/Secret are correct
2. Agent shows "Live" status in Render dashboard
3. Check agent logs for connection errors:
   ```
   ERROR: Failed to connect to LiveKit room
   ```

**Fix**:
- Regenerate LiveKit API keys
- Update environment variables in Render
- Redeploy service

---

### **Issue: RAG System Not Working**

**Check Logs For**:
```
⚠️ Warning: Could not pre-warm RAG system
```

**Fix**:
1. Ensure `documents/` folder exists with content
2. Verify FAISS vector store initialized
3. Check OpenAI API key for embeddings

---

### **Issue: Voice Not Working**

**Verify API Keys**:
- Deepgram API key (STT): Valid and funded
- Cartesia API key (TTS): Valid and funded
- Check logs for 401/403 errors

---

## 🎯 Performance Optimization

### **Free Tier Limitations**

- **Cold Starts**: ~30-60s delay if service sleeps
- **RAM**: 512 MB (sufficient for basic operation)
- **CPU**: 0.1 CPU (may cause slower response times)

### **Upgrade to Starter Plan ($7/month) for**:

- ⚡ No cold starts (always-on instance)
- 🚀 Faster response times (0.5 CPU)
- 📈 Better multi-agent performance
- 🔊 Improved voice processing

---

## 📊 Monitoring & Metrics

Your agent logs performance metrics:

```json
{
  "stt_latency": 0.5,
  "llm_inference": 2.1,
  "tts_generation": 0.3,
  "total_latency": 2.9
}
```

**View in Render Logs**:
- Real-time latency tracking
- Query count
- Error rates
- Usage summaries

---

## 🔄 Auto-Deployment

Your service auto-deploys on git push:

```bash
# Make code changes
git add .
git commit -m "Update agent logic"
git push origin main

# Render automatically:
# 1. Detects push
# 2. Rebuilds Docker image
# 3. Deploys new version
# 4. Zero-downtime rollout
```

---

## 🛡️ Security Best Practices

1. **Never commit `.env.local` to GitHub** ✅
2. **Rotate API keys regularly** (monthly recommended)
3. **Use environment variables** for all secrets ✅
4. **Enable Render's "Auto-Deploy from Protected Branches"**
5. **Monitor usage** to detect anomalies

---

## 📚 Additional Resources

- **LiveKit Agents Docs**: https://docs.livekit.io/agents/
- **Render Docs**: https://docs.render.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Deployment Examples**: https://github.com/livekit-examples/agent-deployment

---

## 🆘 Support

**Issues?**
- Render Support: https://render.com/support
- LiveKit Discord: https://livekit.io/discord
- GitHub Issues: Open issue in your repo

---

## ✅ Deployment Checklist

Before going live:

- [ ] Updated `render.yaml` with correct repo URL
- [ ] Created `ragvoice-ai-env` environment group in Render
- [ ] Added all 6 environment variables (LIVEKIT, OPENAI, DEEPGRAM, CARTESIA, WEATHER)
- [ ] Committed and pushed changes to GitHub
- [ ] Created Blueprint or Private Service in Render
- [ ] Verified deployment logs show "✅ RAG system pre-warmed successfully"
- [ ] Tested voice interaction in LiveKit room
- [ ] Confirmed multi-agent routing works (RAG, Tools, General)
- [ ] Monitored performance metrics

---

**🎉 Congratulations! Your multi-agent voice assistant is now live on Render!**

For questions or issues, check logs first, then consult documentation above.
