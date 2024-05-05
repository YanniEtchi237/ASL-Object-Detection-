from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
    
    # Perform live predictions and show results
    results = model.predict(source="0", show=True)
    
    # Print the results
    print(results)

if __name__ == "__main__":
    main()
