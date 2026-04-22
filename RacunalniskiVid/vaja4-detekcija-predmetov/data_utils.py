import numpy as np
import pandas as pd
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from augmentation import (
    augment, 
    SCALE_RADIUS_MIN, 
    SCALE_RADIUS_MAX, 
    CURRENT_RADIUS, 
    ROTATION_ANGLE_MIN, 
    ROTATION_ANGLE_MAX, 
    TRANSLATION_MIN, 
    TRANSLATION_MAX
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FOLDERS = ['data_2018_09_11', 'data_2018_09_13', 'data_2018_09_14']
TRAIN_VAL_FOLDERS = ['data_2018_09_11', 'data_2018_09_13']
TEST_FOLDER = 'data_2018_09_14'
TRAIN_SPLIT_RATIO = 0.8
DEFAULT_RADIUS = 16


def load_data():
    all_data = []
    
    for folder in FOLDERS:
        csv_path = os.path.join(DATA_DIR, folder, 'label_data.csv')
        labels_df = pd.read_csv(csv_path, sep=' ')
        
        for image_name, group in labels_df.groupby('image'):
            image_path = os.path.join(DATA_DIR, folder, f"{image_name}.png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            labels = {
                'x': group['x'].values,
                'y': group['y'].values,
                'value': group['value'].values - 1,
                'radius': np.full(len(group), DEFAULT_RADIUS)
            }
            
            all_data.append({'image': image, 'labels': labels, 'folder': folder})
    
    return all_data


def split_data(all_data):
    train_val_data = [entry for entry in all_data if entry['folder'] in TRAIN_VAL_FOLDERS]
    test_data = [entry for entry in all_data if entry['folder'] == TEST_FOLDER]
    
    split_index = int(TRAIN_SPLIT_RATIO * len(train_val_data))
    train_data = train_val_data[:split_index]
    validation_data = train_val_data[split_index:]
    
    return train_data, validation_data, test_data


def create_targets(labels, grid_height, grid_width):
    target_classes = torch.full((grid_height, grid_width), 6, dtype=torch.long)
    target_regressions = torch.zeros((3, grid_height, grid_width), dtype=torch.float32)
    hit_mask = torch.zeros((grid_height, grid_width), dtype=torch.float32)

    for i in range(len(labels['x'])):
        grid_x, grid_y = int(labels['x'][i] // 16), int(labels['y'][i] // 16)
        
        if grid_x >= grid_width or grid_y >= grid_height or grid_x < 0 or grid_y < 0:
            continue

        target_classes[grid_y, grid_x] = int(labels['value'][i])
        
        target_regressions[0, grid_y, grid_x] = labels['x'][i] - (grid_x * 16)
        target_regressions[1, grid_y, grid_x] = labels['y'][i] - (grid_y * 16)
        target_regressions[2, grid_y, grid_x] = labels['radius'][i] - 16
        
        hit_mask[grid_y, grid_x] = 1.0

    return target_classes, target_regressions, hit_mask


def batch_collate_fn(batch):
    target_radius = np.random.uniform(SCALE_RADIUS_MIN, SCALE_RADIUS_MAX)
    aug_params = {
        'scale': target_radius / CURRENT_RADIUS,
        'angle': np.random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX),
        'dx': np.random.randint(TRANSLATION_MIN, TRANSLATION_MAX),
        'dy': np.random.randint(TRANSLATION_MIN, TRANSLATION_MAX)
    }

    images, class_targets, regression_targets, masks = [], [], [], []

    for image, labels in batch:
        aug_image, aug_labels = augment(image, labels, aug_params)
        if aug_image is None:
            continue
        
        image_tensor = torch.from_numpy(aug_image).permute(2, 0, 1).float() / 255.0
        
        grid_height, grid_width = image_tensor.shape[1] // 16, image_tensor.shape[2] // 16
        class_target, regression_target, mask = create_targets(aug_labels, grid_height, grid_width)
        
        images.append(image_tensor)
        class_targets.append(class_target)
        regression_targets.append(regression_target)
        masks.append(mask)

    return torch.stack(images), torch.stack(class_targets), torch.stack(regression_targets), torch.stack(masks)


def val_collate_fn(batch):
    images, class_targets, regression_targets, masks = [], [], [], []

    for image, labels in batch:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        grid_height, grid_width = image_tensor.shape[1] // 16, image_tensor.shape[2] // 16
        class_target, regression_target, mask = create_targets(labels, grid_height, grid_width)
        
        images.append(image_tensor)
        class_targets.append(class_target)
        regression_targets.append(regression_target)
        masks.append(mask)

    return torch.stack(images), torch.stack(class_targets), torch.stack(regression_targets), torch.stack(masks)


class DiceDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['image'], self.data[index]['labels']


def get_dataloaders(batch_size=8):
    all_raw_data = load_data()
    train_data, val_data, test_data = split_data(all_raw_data)
    
    train_loader = DataLoader(
        DiceDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_collate_fn
    )
    val_loader = DataLoader(
        DiceDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate_fn
    )
    test_loader = DataLoader(
        DiceDataset(test_data),
        batch_size=1,
        shuffle=False,
        collate_fn=val_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def get_raw_predictions(model, image_tensor, device, confidence_threshold=0.6):
    model.eval()

    with torch.no_grad():
        pred_class_logits, pred_regression = model(image_tensor.unsqueeze(0).to(device))

    pred_class_logits = pred_class_logits[0]
    pred_regression = pred_regression[0]

    probs = torch.softmax(pred_class_logits, dim=0)
    conf, cls_id = torch.max(probs, dim=0)

    valid_mask = (cls_id < 6) & (conf > confidence_threshold)
    gy, gx = torch.where(valid_mask)

    results = []

    for i in range(len(gy)):
        y, x = gy[i], gx[i]
        dx, dy, dr = pred_regression[:, y, x]

        results.append({
            'x': (x * 16 + dx).item(),
            'y': (y * 16 + dy).item(),
            'radius': (16 + dr).item(),
            'value': cls_id[y, x].item(),
            'confidence': conf[y, x].item()
        })

    return results

def extract_ground_truth(class_targets, regression_targets, presence_mask):
    grid_y_indices, grid_x_indices = torch.where(presence_mask == 1.0)

    dx_values = regression_targets[0, grid_y_indices, grid_x_indices]
    dy_values = regression_targets[1, grid_y_indices, grid_x_indices]
    dr_values = regression_targets[2, grid_y_indices, grid_x_indices]

    x_coords = grid_x_indices * 16 + dx_values
    y_coords = grid_y_indices * 16 + dy_values
    radii = 16 + dr_values

    class_values = class_targets[grid_y_indices, grid_x_indices]

    return {
        'x': x_coords.tolist(),
        'y': y_coords.tolist(),
        'radius': radii.tolist(),
        'value': class_values.tolist()
    }
import math

def circle_iou(d1, d2):
    x1, y1, r1 = d1['x'], d1['y'], d1['radius']
    x2, y2, r2 = d2['x'], d2['y'], d2['radius']

    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if d > r1 + r2:
        return 0.0

    if d <= abs(r1 - r2):
        intersection = math.pi * min(r1, r2)**2
    else:
        alpha = math.acos((r1*r1 + d*d - r2*r2) / (2 * r1 * d)) * 2
        beta  = math.acos((r2*r2 + d*d - r1*r1) / (2 * r2 * d)) * 2

        a1 = 0.5 * beta * r2*r2 - 0.5 * r2*r2 * math.sin(beta)
        a2 = 0.5 * alpha * r1*r1 - 0.5 * r1*r1 * math.sin(alpha)

        intersection = a1 + a2

    area1 = math.pi * r1*r1
    area2 = math.pi * r2*r2
    union = area1 + area2 - intersection

    return intersection / union

def remove_overlaps(detections, iou_threshold=0.3):
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)

    kept = []

    for det in detections:
        keep = True

        for k in kept:
            if circle_iou(det, k) > iou_threshold:
                keep = False
                break

        if keep:
            kept.append(det)

    return kept

def to_output_format(detections):
    return {
        'x': [d['x'] for d in detections],
        'y': [d['y'] for d in detections],
        'radius': [d['radius'] for d in detections],
        'value': [d['value'] for d in detections],
        'confidence': [d['confidence'] for d in detections]
    }

def evaluate_image(predictions, ground_truth, iou_threshold=0.5):
    matched_gt = set()

    tp, fp = 0, 0

    for pred in predictions:
        best_iou = 0
        best_idx = -1

        for i in range(len(ground_truth['x'])):
            if i in matched_gt:
                continue

            gt = {
                'x': ground_truth['x'][i],
                'y': ground_truth['y'][i],
                'radius': ground_truth['radius'][i]
            }

            iou = circle_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou > iou_threshold:
            if pred['value'] == ground_truth['value'][best_idx]:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        else:
            fp += 1

    fn = len(ground_truth['x']) - len(matched_gt)

    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1