import numpy as np
import cv2

ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX = -30, 30
SCALE_RADIUS_MIN, SCALE_RADIUS_MAX = 12, 24
CURRENT_RADIUS = 16
TRANSLATION_MIN, TRANSLATION_MAX = -16, 16
DIVISIBLE_BY = 16


def _largest_rect_in_histogram(heights):
    stack = []
    best_area = 0
    best_x = 0
    best_w = 0
    best_h = 0

    for i in range(len(heights) + 1):
        cur_h = heights[i] if i < len(heights) else 0

        while stack and heights[stack[-1]] > cur_h:
            h = heights[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            rect_w = i - left
            area = h * rect_w

            if area > best_area:
                best_area = area
                best_x = left
                best_w = rect_w
                best_h = h

        stack.append(i)

    return best_x, best_w, best_h


def _largest_valid_rect(mask):
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)

    best_area = 0
    best_rect = (0, 0, w, h)

    for y in range(h):
        row = mask[y] > 0
        heights[row] += 1
        heights[~row] = 0

        x, rect_w, rect_h = _largest_rect_in_histogram(heights)
        area = rect_w * rect_h

        if area > best_area:
            best_area = area
            y0 = y - rect_h + 1
            best_rect = (x, y0, rect_w, rect_h)

    return best_rect


def rotate_image_and_labels(img, labels, angle=None):
    if angle is None:
        angle = np.random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)

    h, w = img.shape[:2]

    cx = float(np.mean(labels["x"]))
    cy = float(np.mean(labels["y"]))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = np.full((h, w), 255, dtype=np.uint8)
    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = ((rotated_mask > 250) * 255).astype(np.uint8)

    x0, y0, crop_w, crop_h = _largest_valid_rect(rotated_mask)
    cropped = rotated[y0:y0 + crop_h, x0:x0 + crop_w]

    final_img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    pts = np.stack(
        [labels["x"], labels["y"], np.ones_like(labels["x"])],
        axis=1
    )
    rotated_pts = pts @ M.T

    nx = rotated_pts[:, 0] - x0
    ny = rotated_pts[:, 1] - y0

    scale_x = w / crop_w
    scale_y = h / crop_h

    nx *= scale_x
    ny *= scale_y
    nr = labels["radius"] * ((scale_x + scale_y) * 0.5)

    valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)

    return final_img, {
        "x": nx[valid],
        "y": ny[valid],
        "value": labels["value"][valid],
        "radius": nr[valid],
    }
def scale_image_and_labels(img, labels, scale=None):
    if scale is None:
        scale = np.random.uniform(SCALE_RADIUS_MIN, SCALE_RADIUS_MAX) / CURRENT_RADIUS
    
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return scaled, {
        'x': labels['x'] * scale, 'y': labels['y'] * scale,
        'value': labels['value'].copy(), 'radius': labels['radius'] * scale
    }

def translate_image_and_labels(img, labels, dx=None, dy=None):
    if dx is None: dx = np.random.randint(TRANSLATION_MIN, TRANSLATION_MAX)
    if dy is None: dy = np.random.randint(TRANSLATION_MIN, TRANSLATION_MAX)
    
    h, w = img.shape[:2]
    buffer = 20 
    
    y_s, y_e = max(0, buffer + dy), min(h, h - buffer + dy)
    x_s, x_e = max(0, buffer + dx), min(w, w - buffer + dx)

    cropped = img[y_s:y_e, x_s:x_e]
    nx, ny = labels['x'] - x_s, labels['y'] - y_s
    mask = (nx >= 0) & (nx < cropped.shape[1]) & (ny >= 0) & (ny < cropped.shape[0])
    
    return cropped, {
        'x': nx[mask], 'y': ny[mask], 'value': labels['value'][mask], 'radius': labels['radius'][mask]
    }

def make_divisible_by_16(img, labels):
    h, w = img.shape[:2]
    new_h, new_w = (h // DIVISIBLE_BY) * DIVISIBLE_BY, (w // DIVISIBLE_BY) * DIVISIBLE_BY
    if new_h <= 0 or new_w <= 0: return None, None
    
    y_s, x_s = (h - new_h) // 2, (w - new_w) // 2
    nx, ny = labels['x'] - x_s, labels['y'] - y_s
    mask = (nx >= 0) & (nx < new_w) & (ny >= 0) & (ny < new_h)
    
    return img[y_s:y_s+new_h, x_s:x_s+new_w], {
        'x': nx[mask], 'y': ny[mask], 'value': labels['value'][mask], 'radius': labels['radius'][mask]
    }

def augment(img, labels, params=None):
    while True:
        try:
            np.random.seed() 
            
            p = params if params is not None else {}
            
            img_aug, labels_aug = scale_image_and_labels(img, labels, p.get('scale'))
            img_aug, labels_aug = rotate_image_and_labels(img_aug, labels_aug, p.get('angle'))
            img_aug, labels_aug = translate_image_and_labels(img_aug, labels_aug, p.get('dx'), p.get('dy'))
            
            res_img, res_labels = make_divisible_by_16(img_aug, labels_aug)
            
            if res_img is not None and len(res_labels['x']) > 0:
                return res_img, res_labels
                
        except Exception:
            continue