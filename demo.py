import cv2
from ultralytics import YOLO
import torch

# Load your YOLO model
model = YOLO('epoch50.pt')  # Load the model (no need for map_location here)

# Ensure the model is running on the CPU
model.to('cpu')  # Move model to CPU explicitly

# Print model classes (just for verification)
print(model.names)

# Open the default webcam
webcamera = cv2.VideoCapture(0)

# Set the camera resolution
webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a named window and set it to fullscreen
cv2.namedWindow("Live Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, frame = webcamera.read()
    if not success:
        print("Failed to capture image")
        break

    # Perform tracking/detection using the CPU
    results = model.track(frame, conf=0.4, imgsz=640)  # Higher resolution

    # Extract and annotate results on the frame
    annotated_frame = results[0].plot()  # This returns the frame with plotted results
    cv2.putText(annotated_frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Live Camera", annotated_frame)
    print(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and destroy windows
webcamera.release()
cv2.destroyAllWindows()
