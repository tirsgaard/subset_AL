#from torchdrug import datasets,models, tasks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

from src.dataset_downloader import get_MNIST
# load the dataset
#dataset = datasets.ZINC250k("datasets/ZINC250k")
"""
matplotlib.use('TkAgg')
ds = deeplake.load('hub://activeloop/not-mnist-small')
dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
for data in dataloader:
    not_mnist = data
    print(not_mnist["labels"])
    break

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axs):
    ax.imshow(not_mnist['images'][i])
    #ax.axis('off')
plt.show()
"""
def manifold_classifier(X: np.ndarray) -> np.ndarray:
    """ Function for classifying if the data belongs to the manifold or not. 
    Args:
        X: The data to classify with shape (n_samples, x_dim, x_dim)"""
    
    length = X.shape[-1]
    top_side_brightness = X[..., length//2, :].abs().mean()
    bottom_side_brightness = X[..., length//2+1, :].abs().mean()
    return 1.05*top_side_brightness < bottom_side_brightness
    

train_data, val_data, test_data = get_MNIST(0.8, one_hot=False)
# Apply the classifier to the data
manifold_labels = np.array([manifold_classifier(x) for x, _ in train_data])
# Check correlation between the labels and the classifier
classes_count = torch.zeros(10)
class_manifold_count = torch.zeros(10)
for i, (_, y) in enumerate(train_data):
    classes_count[y] += 1
    class_manifold_count[y] += manifold_labels[i]
    
print("Correlation between class and manifold label")
print((class_manifold_count/classes_count).numpy())
print("Manifold label distribution")
print(manifold_labels.mean())