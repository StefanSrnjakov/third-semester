import matplotlib.pyplot as plt
from augmentation import augment
import numpy as np



N_AUGMENTATIONS = 6

def visualize_sample(img, labels, title, show_confidence=False):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    for i in range(len(labels['x'])):
        x, y = labels['x'][i], labels['y'][i]
        r = labels['radius'][i]
        value = labels['value'][i]
        
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle)

        text = str(value + 1)
        if show_confidence and 'confidence' in labels:
            conf = labels['confidence'][i]
            text += f" ({conf:.2f})"

        plt.text(
            x, y - r - 5,
            text,
            color='black',
            fontsize=14,
            ha='center',
            va='bottom',
            weight='bold'
        )
    
    plt.title(title, fontsize=16, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_augmentations(sample):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(N_AUGMENTATIONS):
        img_aug, labels_aug = augment(sample['image'], sample['labels'])
        if img_aug is not None:
            axes[i].imshow(img_aug)
            for j in range(len(labels_aug['x'])):
                x, y = labels_aug['x'][j], labels_aug['y'][j]
                r = labels_aug['radius'][j]
                value = labels_aug['value'][j]
                circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
                axes[i].add_patch(circle)
                axes[i].text(x, y, str(value + 1), color='white', fontsize=12,
                            ha='center', va='center', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))
            axes[i].set_title(f"Augmentation {i+1}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_batch(images, labels_list):
    batch_size = images.shape[0]
    cols = 4
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        ax = axes[i]
        ax.imshow(img)
        
        labels = labels_list[i]
        for j in range(len(labels['x'])):
            x, y = labels['x'][j], labels['y'][j]
            r = labels['radius'][j]
            val = labels['value'][j]
            
            circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.5)
            ax.add_patch(circle)
            
            ax.text(x, y, str(int(val + 1)), color='white', fontsize=10, 
                    ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
        
        ax.set_title(f"Sample {i} | Shape: {img.shape[:2]}", fontsize=10)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()