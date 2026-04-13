from ultralytics import YOLO
import cv2

# Training is done, so no need for lines 5-6
#model = YOLO("yolo26n.pt")
#model.train(data="data.yaml", epochs=50, imgsz=640)

model = YOLO("weights/best.pt")
cap = cv2.VideoCapture("test.mp4")  # use 0 for webcam, or "yourvideo.mp4"

cv2.namedWindow("CourtMaster", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CourtMaster", 960, 540)  # set your preferred window size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.05)
    annotated = results[0].plot()

    cv2.imshow("CourtMaster", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()