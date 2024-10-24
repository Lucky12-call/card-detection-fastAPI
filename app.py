import cv2
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the YOLO model
model = YOLO('epoch50.pt')
model.to('cpu')  # Ensure model is on CPU

print(model.names)  # Print class names for reference

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform detection
    results = model.track(frame, conf=0.4, imgsz=640)

    # Extract detection information
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],  # Get the name of the detected class
            "confidence": float(box.conf),
            "box": [int(x) for x in box.xyxy[0].tolist()]
        })

    # Return the detection results
    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
