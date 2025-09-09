#!/usr/bin/env python3
"""
Simple Working API Server for EGRU
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="EGRU Simple API")

@app.get("/")
async def root():
    return {"message": "EGRU Simple API is running!", "status": "success"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "EGRU API is working!"}

@app.get("/test")
async def test():
    return {"test": "success", "model": "EGRU Enhanced GRU"}

if __name__ == "__main__":
    print("🚀 Starting Simple EGRU API Server...")
    print("📍 Server will run on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
