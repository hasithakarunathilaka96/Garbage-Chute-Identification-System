import cv2
import supervision as sv
from ultralytics import YOLO


def main():

    # Open the video file for reading
    # Change EVIDENCE_VIDEO_NAME
    cap = cv2.VideoCapture("EVIDENCE_VIDEO_NAME.mp4")                
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc= cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("RESULT_VIDEO_NAME.mp4",fourcc, fps, (width, height), isColor=True)  #Change RESULT_VIDEO_NAME

    # Load the YOLO model
    model = YOLO("yolov8_chute.pt")

    # Create a box annotator to set bounding box parameters
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Perform object detection and Convert detections to the required format
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Extract labels from the model's names for each detection
            labels = [
                f"{model.model.names[class_id]}"
                for _, _, _, class_id, _ 
                in detections
            ]

            # Annotate the frame with bounding boxes and labels
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            # Write the annotated frame to the output video
            out.write(frame)

            if(cv2.waitKey(1) & 0xFF==ord('q')):  # break with 'Q' key
                break
        else:
            break

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()