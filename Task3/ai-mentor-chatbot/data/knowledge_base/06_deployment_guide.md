# Deploying AI Apps on Free Hosting Platforms

## Choosing a Platform
Render, Railway, and PythonAnywhere are common free options for small Flask apps. Render is generally the most beginner-friendly for Flask + API-based AI apps.

## Before You Deploy
1. Ensure requirements.txt lists all dependencies with compatible versions
2. Remove any hardcoded local file paths - use relative paths or environment variables
3. Make sure the app reads the PORT from an environment variable, since hosting platforms assign it dynamically
4. Test locally with a fresh virtual environment to catch missing dependencies

## Handling Cold Starts
Free tier services often "sleep" after a period of inactivity. The first request after sleeping can take 30-60 seconds to wake the server. Document this clearly in your README so reviewers aren't confused by a slow first load.

## Memory Constraints
Free tiers typically cap memory around 512MB. API-based LLMs (like Groq) avoid this problem entirely since the heavy model runs on Groq's servers, not yours - your app only needs to handle lightweight embedding models and API calls.

## Environment Variables on the Platform
Never commit your .env file. Instead, set the same variables (like GROQ_API_KEY) directly in your hosting platform's dashboard under "Environment" settings.
