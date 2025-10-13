from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2 as cv

from compare_patterns import find_best_pattern

app = FastAPI()

@app.post("/match-pattern/")
async def match_pattern(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=404, content={"message": "Invalid image file"})

    best_pattern, best_coord, best_score, img_bytes = find_best_pattern(img)

    return StreamingResponse(img_bytes, media_type="image/jpg", headers={"Pattern": best_pattern, "Coords": str(best_coord), "Score": str(best_score)})
