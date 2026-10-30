from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from lightly.transforms import MoCoV2Transform


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def unnormalize(tensor):
    """
    Assuming they’re normalized, we’ll un-normalize them for display using mean and std from Image Net
    https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t = tensor.clone()
    t = t * std + mean
    t = t.permute(1, 2, 0).numpy()  # CHW → HWC
    t = (t * 255).astype('uint8')
    return t

def visualize(image_path, two_crop_transform):
    """
    Visualize query and key for one example image in image_path.
    """
    img = Image.open(image_path).convert('RGB')

    # Generate two unnormalized crops for visualization
    q, k = two_crop_transform(img)

    q_img = unnormalize(q)
    k_img = unnormalize(k)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(q_img)
    axs[1].set_title("Crop 1 (q)")
    axs[1].axis('off')

    axs[2].imshow(k_img)
    axs[2].set_title("Crop 2 (k)")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # build “base” transform (exactly as in our custom augmentations)
    base_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.9,1.1)),
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(5, translate=(0.05,0.05), scale=(0.95,1.05)),
        T.RandomApply([T.RandomChoice([
            T.RandomAdjustSharpness(0.8),
            T.RandomAdjustSharpness(1.4)
        ])], p=0.2),
        T.RandomApply([T.ColorJitter(0.2,0.2,0.1,0.05)], p=0.5),
        T.RandomGrayscale(0.05),
        T.GaussianBlur(3, sigma=(0.1,2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    if True:
        two_crop = TwoCropsTransform(base_transform)
    else:
        two_crop = MoCoV2Transform()
    visualize("F:\\datasets\\GERALDfinal\\images\\train\\Aachen_Duesseldorf.mp4#t=53.8.jpg", two_crop)
