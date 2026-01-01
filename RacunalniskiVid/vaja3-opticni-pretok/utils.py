
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from IO import readImage, readFlow

FERNEBACK_PARAMS = {
    "pyr_scale": 0.5, # lower -> blurry fast motion
    "levels": 3, # lower -> detect small motions but faster
    "winsize": 15, # lower -> noisy but sharper
    "iterations": 1, # stronger colors (practicall)
    "poly_n": 5, # higher -> smoother motion blobs
    "poly_sigma": 1.0, # higher -> blurry
    "flags": 0 # smoothing using average filter for smoothing
}

def read_flow(path):
    with open(path, 'rb') as f:
        _ = np.fromfile(f, np.float32, count=1)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    return data.reshape(h, w, 2)


def flow_to_rgb(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def epe_loss(pred, target):
    diff = pred - target
    epe = torch.norm(diff, p=2, dim=1)
    return epe.mean()


def center_crop_to_multiple(arr, divisor=64):
    h, w = arr.shape[:2]
    new_h = (h // divisor) * divisor
    new_w = (w // divisor) * divisor

    new_h = max(new_h, divisor) if h >= divisor else h
    new_w = max(new_w, divisor) if w >= divisor else w

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    return arr[top:top + new_h, left:left + new_w, ...]


def load_sample(idx, data_dir="./data"):
    import os
    img1_path = os.path.join(data_dir, f"{idx}_img1.ppm")
    img2_path = os.path.join(data_dir, f"{idx}_img2.ppm")
    flow_path = os.path.join(data_dir, f"{idx}_flow.flo")

    img1 = readImage(img1_path)
    img2 = readImage(img2_path)
    flow = readFlow(flow_path)

    return img1, img2, flow


def visualize_sample(idx, data_dir="./data"):
    img1, img2, flow = load_sample(idx, data_dir)

    flow_bgr = flow_to_rgb(flow)
    flow_rgb = cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(img1)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(img2)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Optical flow (HSV)")
    plt.imshow(flow_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_batch_losses(train_batch_losses, val_batch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_batch_losses, alpha=0.8)
    plt.xlabel("Batch iteration")
    plt.ylabel("EPE Loss")
    plt.title("Training Loss per Batch")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_batch_losses, alpha=0.8)
    plt.xlabel("Validation batch iteration")
    plt.ylabel("EPE Loss")
    plt.title("Validation Loss per Batch")
    plt.grid()
    plt.show()


def plot_avg_train_loss(train_batch_losses, window_size=100):
    n = len(train_batch_losses)
    num_windows = (n + window_size - 1) // window_size

    avg_losses = []
    x_positions = []

    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n)
        window = train_batch_losses[start:end]
        if len(window) == 0:
            continue
        avg_losses.append(np.mean(window))
        center = (start + end - 1) / 2.0
        x_positions.append(center)

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, avg_losses, alpha=0.8)
    plt.xlabel(f"Batch (center of {window_size}-batch window)")
    plt.ylabel("EPE Loss (avg)")
    plt.title(f"Training Loss Averaged over {window_size} Batches")
    plt.grid()
    plt.show()

def evaluate_model_ferneback(test_loader, device, divisor=64, params=FERNEBACK_PARAMS):
    epe_list = []
    
    iteration = 0
    for img1, img2, flow_gt in test_loader:
        iteration += 1
        print(f"iteration {iteration} of {len(test_loader)}")
        if iteration == 30:
            break
        batch_size = img1.size(0)
        
        for i in range(batch_size):
            # Convert to numpy for cropping and processing (C, H, W) -> (H, W, C)
            im1 = img1[i].cpu().numpy().transpose(1, 2, 0)
            im2 = img2[i].cpu().numpy().transpose(1, 2, 0)
            fl = flow_gt[i].cpu().numpy().transpose(1, 2, 0)
            
            # Apply center crop
            im1_crop = center_crop_to_multiple(im1, divisor=divisor)
            im2_crop = center_crop_to_multiple(im2, divisor=divisor)
            fl_crop = center_crop_to_multiple(fl, divisor=divisor)
            
            # Convert to grayscale for Farneback (needs uint8)
            im1_gray = cv2.cvtColor((im1_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor((im2_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Compute Farneback optical flow
            farneback_flow = cv2.calcOpticalFlowFarneback(
                im1_gray, im2_gray, None,
                pyr_scale=params["pyr_scale"], 
                levels=params["levels"],
                winsize=params["winsize"],
                iterations=params["iterations"], 
                poly_n=params["poly_n"], 
                poly_sigma=params["poly_sigma"], 
                flags=params["flags"]
            )
            
            # Compute EPE for this sample
            epe = np.sqrt(np.sum((farneback_flow - fl_crop)**2, axis=2)).mean()
            epe_list.append(epe)
    
    epe_array = np.array(epe_list)
    avg_epe = epe_array.mean()
    return avg_epe, epe_array

def evaluate_model(model, test_loader, device, divisor=64):
    model.eval()
    epe_list = []
    
    with torch.no_grad():
        iteration = 0
        for img1, img2, flow_gt in test_loader:
            iteration += 1
            print(f"iteration {iteration} of {len(test_loader)}")
            # return if 30 batches done
            if iteration == 30:
                break
            batch_size = img1.size(0)
            cropped_img1 = []
            cropped_img2 = []
            cropped_flow = []
            
            for i in range(batch_size):
                im1 = img1[i].cpu().numpy().transpose(1, 2, 0)
                im2 = img2[i].cpu().numpy().transpose(1, 2, 0)
                fl = flow_gt[i].cpu().numpy().transpose(1, 2, 0)
                
                im1_crop = center_crop_to_multiple(im1, divisor=divisor)
                im2_crop = center_crop_to_multiple(im2, divisor=divisor)
                fl_crop = center_crop_to_multiple(fl, divisor=divisor)
                
                cropped_img1.append(torch.from_numpy(im1_crop).permute(2, 0, 1))
                cropped_img2.append(torch.from_numpy(im2_crop).permute(2, 0, 1))
                cropped_flow.append(torch.from_numpy(fl_crop).permute(2, 0, 1))
            
            img1 = torch.stack(cropped_img1).to(device)
            img2 = torch.stack(cropped_img2).to(device)
            flow_gt = torch.stack(cropped_flow).to(device)
            
            x = torch.cat([img1, img2], dim=1)
            pred_flow = model(x)
            
            # Compute EPE for each image in the batch
            for i in range(batch_size):
                pred_flow_np = pred_flow[i].cpu().numpy().transpose(1, 2, 0)
                flow_gt_np = flow_gt[i].cpu().numpy().transpose(1, 2, 0)
                epe = np.sqrt(np.sum((pred_flow_np - flow_gt_np)**2, axis=2)).mean()
                epe_list.append(epe)
    
    epe_array = np.array(epe_list)
    avg_epe = epe_array.mean()
    return avg_epe, epe_array


def visualize_val_sample_from_paths(
    model,
    img1_path,
    img2_path,
    flow_path=None,
    device="cuda",
    divisor=64,
    params=FERNEBACK_PARAMS
):
    model = model.to(device)
    model.eval()

    img1 = readImage(img1_path).astype(np.float32) / 255.0
    img2 = readImage(img2_path).astype(np.float32) / 255.0

    if flow_path is not None:
        flow = readFlow(flow_path).astype(np.float32)
    else:
        flow = None

    img1 = center_crop_to_multiple(img1, divisor=divisor)
    img2 = center_crop_to_multiple(img2, divisor=divisor)
    if flow is not None:
        flow = center_crop_to_multiple(flow, divisor=divisor)

    img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    farneback_flow = cv2.calcOpticalFlowFarneback(
        img1_gray, img2_gray, None,
        pyr_scale=params["pyr_scale"], levels=params["levels"],winsize=params["winsize"],
        iterations=params["iterations"], poly_n=params["poly_n"], poly_sigma=params["poly_sigma"], flags=params["flags"]
    )
    farneback_flow_vis = flow_to_rgb(farneback_flow)

    img1_t = torch.from_numpy(img1).permute(2, 0, 1)
    img2_t = torch.from_numpy(img2).permute(2, 0, 1)

    if flow is not None:
        flow_t = torch.from_numpy(flow).permute(2, 0, 1)

    img1_t = img1_t.unsqueeze(0).to(device)
    img2_t = img2_t.unsqueeze(0).to(device)

    if flow is not None:
        flow_t = flow_t.unsqueeze(0).to(device)

    with torch.no_grad():
        x = torch.cat([img1_t, img2_t], dim=1)
        pred_flow = model(x)

    # Move to CPU for plotting
    im1 = img1_t[0].cpu().numpy().transpose(1, 2, 0)
    im2 = img2_t[0].cpu().numpy().transpose(1, 2, 0)
    pred_flow_np = pred_flow[0].cpu().numpy().transpose(1, 2, 0)

    if flow is not None:
        gt_flow_np = flow_t[0].cpu().numpy().transpose(1, 2, 0)
        gt_flow_vis = flow_to_rgb(gt_flow_np)
    else:
        gt_flow_vis = None

    pred_flow_vis = flow_to_rgb(pred_flow_np)

    # Compute EPE (End-Point Error) if GT is available
    if flow is not None:
        farneback_epe = np.sqrt(np.sum((farneback_flow - gt_flow_np)**2, axis=2)).mean()
        pred_epe = np.sqrt(np.sum((pred_flow_np - gt_flow_np)**2, axis=2)).mean()
    else:
        farneback_epe = None
        pred_epe = None

    # Plot with 5 columns
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 5, 1)
    plt.title("Image 1")
    plt.imshow(im1)
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.title("Image 2")
    plt.imshow(im2)
    plt.axis("off")

    plt.subplot(1, 5, 3)
    if gt_flow_vis is not None:
        plt.title("GT Flow")
        plt.imshow(gt_flow_vis)
    else:
        plt.title("GT Flow (not provided)")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    if farneback_epe is not None:
        plt.title(f"Farneback (OpenCV)\nEPE: {farneback_epe:.3f}")
    else:
        plt.title("Farneback (OpenCV)")
    plt.imshow(farneback_flow_vis)
    plt.axis("off")

    plt.subplot(1, 5, 5)
    if pred_epe is not None:
        plt.title(f"Predicted Flow (FlowNet)\nEPE: {pred_epe:.3f}")
    else:
        plt.title("Predicted Flow (FlowNet)")
    plt.imshow(pred_flow_vis)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return farneback_epe, pred_epe



