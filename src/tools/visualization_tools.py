import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color
from src.tools.model_tools import get_k_similar_group
import torch

def visualize_samples(imgs_path, gray=False, n_cols=5, n_rows=1):
    """Visualize samples."""

    plt.figure(figsize = (3*n_cols,3*n_rows))
    for n,i in enumerate(np.random.randint(len(imgs_path), size = n_cols*n_rows)):
        plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        img = io.imread(imgs_path[i])
        if gray:
            img = color.rgb2gray(img)
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
    plt.show()

def visualize_torch(images, gray=False, n_cols=5, n_rows=1, caption=""):
    """Visualize samples."""

    fig = plt.figure(figsize = (3*n_cols,3*n_rows))
    for i in range(n_cols*n_rows):
        plt.subplot(n_rows,n_cols,i+1)
        plt.axis('off')
        img = images[i].permute(1, 2, 0).squeeze()
        if gray:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
    fig.text(0, 0, caption)
    return fig

def visualize_group_loss(model, dataloader):
    for x, y, indices, distances in get_k_similar_group(model, loader=dataloader, k=3):
        print(x.shape)
        print(indices.shape)
        for img_idx, closest_idx in enumerate(indices):
            if img_idx % 3 == 0:
                img_matches = torch.stack([ x[int(img_idx)], x[int(closest_idx[0])], x[int(closest_idx[1])], x[int(closest_idx[2])] ])
                caption = f'labels_{img_idx}: ' + str(y[int(img_idx)].item())
                for i in range(len(closest_idx)):
                    caption = caption+' '+str(y[int(closest_idx[i])].item())+' '+'distance: {:.3f}'.format(distances[img_idx, closest_idx[i]]) + " | "
                visualize_torch(img_matches, n_cols=4, n_rows=1, caption=caption)
                if int(img_idx) == 15:
                    break
        break