import pafy
import cv2
import torch
from torchvision import transforms
from model import ResnetModel, EffnetModel
from face_module import get_face_coords
from meter import Meter
from utils import download_weights

# model wieghts url
effb0_net_url = 'https://github.com/yMayanand/Emotion-Recognition/releases/download/v1.0.0/eff_b0.pt'
res18_net_url = 'https://github.com/yMayanand/Emotion-Recognition/releases/download/v1.0.0/res18.pt'

# statistics of imagenet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transforms for image
val_transform = transforms.Compose([
		transforms.ToTensor(),
        transforms.Resize((48, 48)),
		transforms.Normalize(mean, std)
	])

# model for emotion classification
model = EffnetModel()
# loading pretrained model
fname = download_weights(effb0_net_url)
state_dict = torch.load(fname)
model.load_state_dict(state_dict['weights'])

# emotion classes
emotions = ['neutral', 'happy :-)', 'surprise :-O', 'sad', 'angry >:(', 
            "disgust D-':", 'fear', 'contempt', 'unknown', 'NF']

# colors for text for each emotion classes
colors = [(0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 191, 0), (0, 0, 255),
          (255, 255, 0), (0, 191, 255), (255, 0, 191), (255, 0, 191), (255, 0, 191)]

# youtube url
url = "https://www.youtube.com/watch?v=B0ouAnmsO1Y"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

# reading youtube stream
cap = cv2.VideoCapture('/home/thinkin-machine/VS Code Workspaces/FER/video.mp4')

# count to check if meter is initialised or not
CHECK_FLAG = False

# saving video
result = cv2.VideoWriter('result.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         2, (600, 600))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    h, w, c = image.shape

    # initialising meter
    if not CHECK_FLAG:
        # meter
        m = Meter((w//2, h), w//4, (255, 0, 0))
        CHECK_FLAG = True

    # storing orignal image in bgr mode
    orig_image = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    coords = get_face_coords(image)
    if coords:
        # getting bounding box coordinates for face
        xmin, ymin, xmax, ymax = coords
        model.eval()

        image = image[ymin:ymax, xmin:xmax, :]

        # check if face detected is not on edge of the screen
        h, w, c = image.shape
        if not (h and w):
            idx = 9
            continue

        image_tensor = val_transform(image).unsqueeze(0)
        out = model(image_tensor)

        # prediction emotion for detected face
        pred = torch.argmax(out, dim=1)
        idx = pred.item()
        pred_emot = emotions[pred.item()]
        color = colors[idx]

        # drawing annotations on orignal bgr image
        orig_image = cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    else:
        idx = 9
        pred_emot = 'Face Not Detected'
        color = colors[-1]

    """orig_image = cv2.putText(
        orig_image, pred_emot, (10, 30), 
        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, 
        cv2.LINE_AA
    )"""

    m.draw_meter(orig_image, idx)
    result.write(orig_image)

    cv2.imshow('Emotion Detection', orig_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
result.release()
cap.release()
cv2.destroyAllWindows()

