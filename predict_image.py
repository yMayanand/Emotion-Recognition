import cv2
import torch
from torchvision import transforms
from model import ResnetModel, EffnetModel
from face_module import get_face_coords
from meter import Meter
from utils import download_weights

# statistics of imagenet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# model wieghts url
effb0_net_url = 'https://github.com/yMayanand/Emotion-Recognition/releases/download/v1.0.0/eff_b0.pt'
res18_net_url = 'https://github.com/yMayanand/Emotion-Recognition/releases/download/v1.0.0/res18.pt'

# transforms for image
val_transform = transforms.Compose([
		transforms.ToTensor(),
        transforms.Resize((48, 48)),
		transforms.Normalize(mean, std)
	])

def load_model(model_name):

    # model for emotion classification
    if model_name == 'effb0':
        model = EffnetModel() 
        fname = download_weights(effb0_net_url)
    elif model_name == 'res18':
        model = ResnetModel
        fname = download_weights(res18_net_url)
    else:
        raise ValueError('Enter correct model_name')

    # loading pretrained model
    state_dict = torch.load(fname)
    model.load_state_dict(state_dict['weights'])
    return model

# emotion classes
emotions = ['neutral', 'happy :-)', 'surprise :-O', 'sad', 'angry >:(', 
            "disgust D-':", 'fear', 'contempt', 'unknown', 'NF']

# colors for text for each emotion classes
colors = [(0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 191, 0), (0, 0, 255),
          (255, 255, 0), (0, 191, 255), (255, 0, 191), (255, 0, 191), (255, 0, 191)]

def predict(image, save_path):
    image = cv2.imread(image)
    h, w, c = image.shape

    # meter
    m = Meter((w//2, h), w//5, (255, 0, 0))

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

    orig_image = cv2.flip(orig_image, 1)
    m.draw_meter(orig_image, idx)

    cv2.imwrite(save_path, orig_image)
    return orig_image

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='path to image location')
    parser.add_argument('--model_name', type=str, default='effb0', help='name of the model')
    parser.add_argument('--save_path', type=str, default='./result.jpg', help='path to save image')
    args = parser.parse_args()
    model = load_model(args.model_name)

    predict(args.image_path, args.save_path)
