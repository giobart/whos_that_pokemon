import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color

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