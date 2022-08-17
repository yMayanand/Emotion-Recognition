import cv2
import torch

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

def freeze(model, unfreeze=False):
    """Function to freeze model parameters necessary for finetuning.
    """
    for param in model.parameters():
        param.requires_grad = unfreeze
        
def accuracy(predictions, ground_truth):
    """Funtion to calculate accuracy of the model.
    """
    
    _, preds = torch.max(predictions, dim=1)
    score = (preds == ground_truth).float().mean()
    return score.item()
