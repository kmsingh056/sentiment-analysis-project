from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pandas as pd
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-trained sentiment analysis model (BERT)
sentiment_analyzer = pipeline("sentiment-analysis")

# Authentication setup
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Analyze CSV endpoint
@app.post("/analyze_csv/")
async def analyze_csv(file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)

    # Validate file format
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        # Read the uploaded file into a DataFrame
        data = pd.read_csv(file.file)

        # Validate required columns
        if "id" not in data.columns or "text" not in data.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'id' and 'text' columns.")
        
        # Perform sentiment analysis
        results = []
        for _, row in data.iterrows():
            analysis = sentiment_analyzer(row["text"])
            results.append({
                "id": row["id"],
                "text": row["text"],
                "sentiment": analysis[0]["label"],
                "score": analysis[0]["score"]
            })
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")