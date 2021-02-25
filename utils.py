import torch
import numpy as np

from PIL import Image

MEAN = torch.tensor([[[0.485, 0.456, 0.406]]]).T
STD = torch.tensor([[[0.229, 0.224, 0.225]]]).T

def load_image(image, crop=None, ratio="full"):
    """
    Args:
        image : string, path to image
        crop : the crop rectangle, as a (left, upper, right, lower)-tuple
        ratio : ["full", "min"]
    """

    assert ratio in ["full", "min"], "ratio arg supports only full or min"
    
    img = Image.open(image)
    
    if crop:
        img = img.crop(box=crop)

    if ratio == "min":
        img = img.resize((224,224))

    img_torch = torch.from_numpy(np.array(img)).permute(2,0,1)
    img_torch = img_torch/255
    img_torch = (img_torch - MEAN)/STD

    return img, img_torch

def total_variation(img, epsilon=0.0001):
    dx = torch.roll(img, shifts=(0, 1), dims=(-1, -2)) - img #cheap gradient 
    dy = torch.roll(img, shifts=(1, 0), dims=(-1, -2)) - img
    return torch.sqrt(dx**2 + dy**2  + epsilon).mean()  #regularized TV

def G(array):
    """Flatten array from last dim.
    Take first element from list of batch, because
    array's shape is torch.Size([1, 256, 28, 28])
    """
    return array[0].flatten(start_dim=1)


"""
img_rescaled = ((img_result - img_result.min())/(img_result.max() - img_result.min())) * 255
Image.fromarray(
    img_rescaled.detach().numpy().astype("uint8")
)
"""