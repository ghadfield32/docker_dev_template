from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ML API",
    description="Machine Learning API for React Frontend",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs"
)

# Updated CORS configuration for React frontend
origins = [
    "http://localhost:3000",  # React development server
    "http://localhost:3001",  # Alternative React port
    "https://localhost:3000", # HTTPS version
    "https://your-react-app.vercel.app",  # Replace with your React deployment URL
    "https://your-react-app.netlify.app", # Replace with your React deployment URL
]

# Add environment variable for production origins
if os.getenv("FRONTEND_URL"):
    origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# Add a root endpoint for testing
@app.get("/")
async def root():
    return {"message": "ML API is running", "docs": "/api/v1/docs"}

# Add a simple test endpoint for ML functionality
@app.get("/api/v1/test")
async def test_endpoint():
    return {"message": "Test endpoint working", "status": "success"}

# Add a simple prediction endpoint placeholder
@app.post("/api/v1/predict")
async def predict(data: dict):
    return {"prediction": "placeholder", "input": data, "status": "success"}
