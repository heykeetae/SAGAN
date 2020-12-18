import torch
import numpy as np
# import matlibplot.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def Logger(path:str):
    logger = SummaryWriter(path)
    return logger

def show_img(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    return npimg
def image_grid_writer(writer, data, name, step, nrow=8):
    image_grid = make_grid(data, nrow=nrow)
    image_grid = show_img(image_grid)
    writer.add_image(name, image_grid,step)