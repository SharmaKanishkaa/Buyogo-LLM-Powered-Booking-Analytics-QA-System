from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from HotelBookingPipeline import HotelBookingPipeline
from HotelBookingRAG import HotelBookingRAG
from models import init_db, SessionLocal, HotelBooking, QueryHistory
from sqlalchemy.orm import Session
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Hotel Booking Analytics API",
    description="API for analyzing hotel booking data with RAG capabilities",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class AnalyticsRequest(BaseModel):
    filters: Optional[dict] = None
    include_visualizations: bool = True

class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = False

# Initialize systems
DATA_PATH = "C:/Users/Kanishka/Documents/Buyogo/hotel_bookings.csv"
pipeline = HotelBookingPipeline(DATA_PATH)
analytics_data = pipeline.run_pipeline()
rag_system = HotelBookingRAG(analytics_data)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Booking RAG system!"}

@app.on_event("startup")
async def startup_event():
    init_db()
    rag_system.query("What is the average daily rate?")

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest, db: Session = Depends(get_db)):
    try:
        # Optionally fetch real-time stats from DB here
        response = {
            "summary_stats": analytics_data["summary_stats"],
            "monthly_metrics": analytics_data["monthly_metrics"],
            "cancellation_analysis": analytics_data["cancellation_analysis"],
            "top_countries": analytics_data["top_countries"]
        }

        if request.include_visualizations:
            response["visualizations"] = {
                "monthly_adr": "http://localhost:8000/visualizations/monthly_adr.png",
                "cancellation_by_country": "http://localhost:8000/visualizations/cancellation_by_country.png"
            }

        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def answer_question(request: QuestionRequest, db: Session = Depends(get_db)):
    try:
        result = rag_system.query(request.question)
        response = {"answer": result["answer"]}

        # Save query to DB
        history = QueryHistory(
            question=request.question,
            answer=result["answer"],
            timestamp=datetime.utcnow()
        )
        db.add(history)
        db.commit()

        if request.include_sources:
            response["sources"] = result["sources"]
            response["metadata"] = result["metadata"]

        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/{viz_name}")
async def get_visualization(viz_name: str):
    valid_viz = ["monthly_adr", "cancellation_by_country"]
    if viz_name not in valid_viz:
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    file_path = f"static/visualizations/{viz_name}.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not generated")
    
    return FileResponse(file_path)

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # Try DB read
        db.query(HotelBooking).first()
        db_status = "online"
    except:
        db_status = "offline"
    
    return {
        "status": "healthy",
        "components": {
            "pipeline": "operational",
            "rag": "operational",
            "vector_db": "ready" if rag_system.vector_db else "offline",
            "database": db_status
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='127.0.0.1')

# Run: uvicorn main:app --reload
