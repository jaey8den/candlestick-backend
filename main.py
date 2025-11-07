from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2 as cv
import uuid
import asyncio
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from compare_patterns import find_best_pattern

app = FastAPI()

# Limit which domain can access the api
origins = [
    "https://candlestick-matcher.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks : Dict[str, dict] = {}

executor = ThreadPoolExecutor(max_workers=2)

# Run processing seperately to not block other requests like polling
def process_image_sync(img):
    print("Entered executor...")

    best_pattern, best_score, img_base64 = find_best_pattern(img)
    
    return {
        "pattern": best_pattern,
        "score": best_score,
        "image": img_base64
    }

async def process_image_task(task_id: str, img):
    """Background task for image processing"""
    try:
        print("Entered background task...")
        tasks[task_id]["status"] = "processing"
        
        loop = asyncio.get_event_loop()
        print("Loop created. Entering loop...")
        result = await loop.run_in_executor(
            executor,
            process_image_sync,
            img
        )
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        
        print("Completed background tasks.")
    except Exception as e:
        print("error:", e)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.post("/match-pattern/")
async def match_pattern(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        task_id = str(uuid.uuid4())

        # Initialize task
        tasks[task_id] = {
            "status": "queued",
            "result": None,
            "error": None
        }

        print("Received file:", file.filename)
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        print("Decoding image...")      
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=404, detail="Invalid image file")

        background_tasks.add_task(process_image_task, task_id, img)

        print("Returning Task ID.")
        return JSONResponse({"task_id": task_id})
    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check processing status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]

@app.get("/healthcheck/")
async def healthcheck():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "hi there"}