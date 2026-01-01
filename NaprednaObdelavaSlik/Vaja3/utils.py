import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
from dataset import TextSegmentationDataset, LARGE_SIZE


def visualize_dataset_sample(dataset: TextSegmentationDataset, idx: int = 0, num_samples: int = 3):
    """Visualize samples from the dataset."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        image_tensor, mask_tensor = dataset[idx]
        
        # Convert tensors back to numpy for visualization
        image = image_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
        mask = mask_tensor.permute(1, 2, 0).numpy()  # [H, W, 1]
        mask = mask.squeeze()  # [H, W]
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()


def demonstrate_image_preparation(dataset: TextSegmentationDataset):
    # Load random images
    bg_path = random.choice(dataset.image_paths)
    bg_img = np.array(Image.open(bg_path).convert("RGB"))
    
    texture_path = random.choice(dataset.image_paths)
    while texture_path == bg_path:
        texture_path = random.choice(dataset.image_paths)
    texture_img = np.array(Image.open(texture_path).convert("RGB"))
    
    # Step 1: Random crop background
    image_bg = dataset._random_crop(bg_img, LARGE_SIZE)
    
    # Step 2: Random crop texture
    image_text = dataset._random_crop(texture_img, LARGE_SIZE)
    
    # Step 3: Generate text mask
    mask, font_info = dataset.generate_text_mask(LARGE_SIZE)
    
    # Step 4: Apply texture to background
    image_combined = dataset.apply_texture_to_background(image_bg, image_text, mask)
    
    # Step 5: Apply augmentation
    image_uint8 = (image_combined * 255.0).astype(np.uint8)
    mask_uint8 = (mask * 255.0).astype(np.uint8)
    image_aug, mask_aug = dataset._apply_augmentation(image_uint8, mask_uint8, augment=True)
    image_aug = image_aug.astype(np.float32) / 255.0
    mask_aug = mask_aug.astype(np.float32) / 255.0
    
    # Visualize all steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image_bg.astype(np.uint8))
    axes[0, 0].set_title("Step 1: Background (cropped)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(image_text.astype(np.uint8))
    axes[0, 1].set_title("Step 2: Texture (cropped)")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(mask.squeeze(), cmap="gray")
    axes[0, 2].set_title("Step 3: Text Mask")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(image_combined)
    axes[1, 0].set_title("Step 4: Combined (with texture)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(image_aug)
    axes[1, 1].set_title("Step 5: After Augmentation")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(mask_aug.squeeze(), cmap="gray")
    axes[1, 2].set_title("Final Mask")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Font info: {font_info}")


def demonstrate_model_with_1_image(model, dataset, device, image_path=None):
    if image_path is not None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img = Image.open(image_path).convert("RGB")
        test_image = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(test_image)
            prediction = torch.sigmoid(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(test_image[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(prediction[0].squeeze().cpu().numpy(), cmap="gray")
        axes[1].set_title("Prediction")
        axes[1].axis("off")
    else:
        test_image, test_mask = dataset[0]
        test_image = test_image.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(test_image)
            prediction = torch.sigmoid(prediction)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(test_image[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(test_mask.squeeze().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(prediction[0].squeeze().cpu().numpy(), cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def test_model_metrics(model, dataset, device, num_test_images=1000):
    iou_scores = []
    dice_scores = []

    print(f"Testing on {num_test_images} images...")
    for i in range(num_test_images):
        image, mask = dataset[0]
        image = image.unsqueeze(0).to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            prediction = model(image)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()
        
        prediction = prediction.squeeze()
        mask = mask.squeeze()
        
        TP = (prediction * mask).sum().item()
        FP = (prediction * (1 - mask)).sum().item()
        FN = ((1 - prediction) * mask).sum().item()
        
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        iou_scores.append(iou)
        
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        dice_scores.append(dice)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_test_images} images")

    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)

    print(f"\nResults on {num_test_images} test images:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice coefficient: {avg_dice:.4f}")
    
    return avg_iou, avg_dice

