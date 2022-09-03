import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def get_face_coords(image):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(image)
        # Draw face detections of each face.
        if not results.detections:
            return False

        # shape of image
        h, w, _ = image.shape
        
        t = results.detections[0].location_data.relative_bounding_box
        height = t.height * h
        ymin = t.ymin * h
        width = t.width * w
        xmin = t.xmin * w
        xmax = xmin + width
        ymax = ymin + height
        return int(xmin), int(ymin), int(xmax), int(ymax)