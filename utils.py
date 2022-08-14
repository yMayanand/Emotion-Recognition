import cv2

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