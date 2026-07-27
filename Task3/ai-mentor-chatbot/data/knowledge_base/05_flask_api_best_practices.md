# Flask API Best Practices for ML/NLP Apps

## Load Models Once
Load your trained model or vector store once at server startup, not on every request. Reloading per-request is slow and can exhaust memory on free hosting tiers.

## Structure Your Endpoints
- Use a dedicated endpoint for the core function (e.g. /respond or /recommend)
- Include a /health endpoint that returns a simple status like {"status": "ok"} to confirm the server is running

## Input Validation
Always check that required fields exist in the request JSON before processing. Return a 400 status with a clear error message for missing or malformed input, rather than letting the app throw an unhandled exception.

## Error Handling Pattern
Wrap model inference calls in try/except blocks. Catch specific exceptions where possible, log the error server-side, and return a generic but clear JSON error message to the client - never expose stack traces or internal details in the API response.

## CORS
If your frontend and backend are served separately (or the frontend calls the API via JavaScript fetch), enable CORS using flask-cors so browser requests aren't blocked.

## Environment Variables
Store API keys and secrets in a .env file, loaded via python-dotenv. Never hardcode credentials in source files.
