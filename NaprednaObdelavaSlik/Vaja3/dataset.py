from PIL import Image
import numpy as np
import random
import os
import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import cv2 as cv
import string

IMAGE_SIZE = 256  # Final image size after augmentation
LARGE_SIZE = 512  # Size before augmentation (to avoid black borders)


class TextSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = True):
        self.root_dir = root_dir
        self.augment = augment
        
        exts = ("*.png",)
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        
        self.image_paths = sorted(image_paths)
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
        
        # Hershey font family (one font per image, same for all rows)
        self.fonts = [
            cv.FONT_HERSHEY_SIMPLEX,
            cv.FONT_HERSHEY_PLAIN,
            cv.FONT_HERSHEY_DUPLEX,
            cv.FONT_HERSHEY_COMPLEX,
            cv.FONT_HERSHEY_TRIPLEX,
        ]
        
        # Characters for random text generation
        self.chars = string.ascii_letters + string.digits + " "
    
    def __len__(self):
        return 10**9
    
    def _generate_random_text(self, min_length: int = 3, max_length: int = 15) -> str:
        length = random.randint(min_length, max_length)
        chars_with_spaces = string.ascii_letters + string.digits + " " * 5
        return ''.join(random.choice(chars_with_spaces) for _ in range(length))
    
    def _random_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h < size or w < size:
            scale = max(size / h, size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            h, w = img.shape[:2]
        
        if h < size or w < size:
            img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
            return img
        
        if h == size:
            y = 0
        else:
            y = random.randint(0, max(0, h - size))
        
        if w == size:
            x = 0
        else:
            x = random.randint(0, max(0, w - size))
        
        cropped = img[y:y+size, x:x+size]
        
        if cropped.shape[0] != size or cropped.shape[1] != size:
            cropped = cv.resize(cropped, (size, size), interpolation=cv.INTER_LINEAR)
        
        return cropped
    
    def generate_text_mask(self, size: int = LARGE_SIZE) -> Tuple[np.ndarray, dict]:
        # Create mask (black image)
        target = np.zeros((size, size, 3), dtype=np.uint8)
        
        # One font per image (same for all rows)
        font_face = random.choice(self.fonts)
        font_scale = random.uniform(0.8, 2.0)
        thickness = random.randint(1, 3)
        
        font_info = {
            'font_face': font_face,
            'font_scale': font_scale,
            'thickness': thickness
        }
        
        # Calculate text height to determine how many rows we need
        sample_text = "Sample"
        (_, text_h), baseline = cv.getTextSize(sample_text, font_face, font_scale, thickness)
        row_spacing = int(text_h * 1.2)
        
        # Calculate number of rows needed to fill the image
        start_y = 20
        available_height = size - start_y - 20
        num_rows = available_height // (text_h + row_spacing)
        
        texts = [self._generate_random_text() for _ in range(num_rows)]
        
        # Draw each row of text to fill the image
        current_y = start_y
        for text in texts:
            (text_w, text_h), baseline = cv.getTextSize(text, font_face, font_scale, thickness)
            
            max_x = max(10, size - text_w - 10)
            x = random.randint(10, max_x)
            position = (x, current_y)
            
            target = cv.putText(target, text, position, font_face, font_scale, 
                              [255, 255, 255], thickness, cv.LINE_8)
            
            current_y += text_h + row_spacing
            
            if current_y >= size - 20:
                break
        
        target = target[:, :, 0:1].astype(np.float32) / 255.0
        
        return target, font_info
    
    def apply_texture_to_background(self, image_bg: np.ndarray, image_text: np.ndarray, 
                                   mask: np.ndarray) -> np.ndarray:
        h_mask, w_mask = mask.shape[:2]
        
        if image_bg.shape[:2] != (h_mask, w_mask):
            image_bg = cv.resize(image_bg, (w_mask, h_mask), interpolation=cv.INTER_LINEAR)
        if image_text.shape[:2] != (h_mask, w_mask):
            image_text = cv.resize(image_text, (w_mask, h_mask), interpolation=cv.INTER_LINEAR)
        
        image_bg = image_bg.astype(np.float32)
        image_text = image_text.astype(np.float32)
        image_combined = (1 - mask) * image_bg + mask * image_text
        image_combined = image_combined.astype(np.float32) / 255.0
        return image_combined
    
    def prepare_synthetic_image(self, bg_img: np.ndarray, texture_img: np.ndarray, 
                               augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Random crop background to LARGE_SIZE
        image_bg = self._random_crop(bg_img, LARGE_SIZE)
        
        # Random crop texture to LARGE_SIZE
        image_text = self._random_crop(texture_img, LARGE_SIZE)
        
        # Generate text mask
        mask, _ = self.generate_text_mask(LARGE_SIZE)
        
        # Apply texture interpolation
        image_combined = self.apply_texture_to_background(image_bg, image_text, mask)
        
        # Convert to uint8 for augmentation
        image_uint8 = (image_combined * 255.0).astype(np.uint8)
        mask_uint8 = (mask * 255.0).astype(np.uint8)
        
        # Apply augmentation
        image_aug, mask_aug = self._apply_augmentation(image_uint8, mask_uint8, augment)
        
        # Convert back to float32 and normalize
        image_aug = image_aug.astype(np.float32) / 255.0
        mask_aug = mask_aug.astype(np.float32) / 255.0
        
        # Ensure mask_aug is 3D (H, W, 1) - augmentation might return 2D
        if len(mask_aug.shape) == 2:
            mask_aug = mask_aug[:, :, np.newaxis]
        
        return image_aug, mask_aug
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray, augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Store original mask shape to preserve it
        mask_was_3d = len(mask.shape) == 3
        
        if not augment:
            # Just crop to final size
            h, w = image.shape[:2]
            y = (h - IMAGE_SIZE) // 2
            x = (w - IMAGE_SIZE) // 2
            image_crop = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
            mask_crop = mask[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
            # Ensure mask maintains 3D shape if it was 3D
            if mask_was_3d and len(mask_crop.shape) == 2:
                mask_crop = mask_crop[:, :, np.newaxis]
            return image_crop, mask_crop
        
        # Random rotation (larger range for more variation)
        angle = random.uniform(-90, 90)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
        mask = cv.warpAffine(mask, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
        
        # Random scaling
        scale = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv.flip(image, 1)
            mask = cv.flip(mask, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = cv.flip(image, 0)
            mask = cv.flip(mask, 0)
        
        # Crop to final size (centered)
        h, w = image.shape[:2]
        y = (h - IMAGE_SIZE) // 2
        x = (w - IMAGE_SIZE) // 2
        image = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
        mask = mask[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
        
        # Ensure mask maintains 3D shape if it was 3D
        if mask_was_3d and len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        return image, mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load random background image
        bg_path = random.choice(self.image_paths)
        bg_img = np.array(Image.open(bg_path).convert("RGB"))
        
        # Load random texture image (different from background)
        texture_path = random.choice(self.image_paths)
        while texture_path == bg_path:
            texture_path = random.choice(self.image_paths)
        texture_img = np.array(Image.open(texture_path).convert("RGB"))
        
        # Prepare synthetic image using the helper function
        image_aug, mask_aug = self.prepare_synthetic_image(bg_img, texture_img, augment=self.augment)
        
        # Convert to tensors: [C, H, W] format
        image_tensor = torch.from_numpy(image_aug).permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask_aug).permute(2, 0, 1)  # [1, H, W]
        
        return image_tensor, mask_tensor

