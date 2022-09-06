import cv2
import torch
from urllib.request import urlretrieve

def read_image(file):
    """Reads the image file

    Returns the numpy array.

    Args:
        file : path to the image

    Returns:
        (numpy.ndarray): image read as numpy array
    """
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
        
def accuracy(predictions, ground_truth):
    """Funtion to calculate accuracy of the model.
    """
    
    _, preds = torch.max(predictions, dim=1)
    score = (preds == ground_truth).float().mean()
    return score.item()

def download_weights(url):
    fname = url.split('/')[-1]
    urlretrieve(url, url.split('/')[-1])
    return fname