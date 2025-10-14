# train_edge_preserving_denoiser.py
import os, re, glob, math, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utils
# ---------------------------
def set_seed(s=123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = arr.transpose(2,0,1)  # HWC -> CHW
    return torch.from_numpy(arr)

def tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
    return Image.fromarray((t*255+0.5).astype(np.uint8))

def psnr(x: torch.Tensor, y: torch.Tensor, eps=1e-8) -> float:
    # x,y: [C,H,W] in [0,1]
    mse = F.mse_loss(x, y).item()
    if mse <= eps: return 99.0
    return 10.0 * math.log10(1.0 / mse)

# ---------------------------
# Dataset
# ---------------------------
class NoisyCleanDataset(Dataset):
    """
    root/{clean, additive, multiplicative}
      clean:           img_0000.png
      additive:        img_0000_add_015.png
      multiplicative:  image_0000_mul_0.11.png  (or img_{id}_mul_*.png)
    """
    def __init__(self, root: str, size: Optional[int]=None):
        self.root = Path(root)
        self.clean_dir = self.root / "clean"
        self.add_dir = self.root / "additive"
        self.mul_dir  = self.root / "multiplicative"
        assert self.clean_dir.is_dir() and self.add_dir.is_dir() and self.mul_dir.is_dir(), "Folders missing"

        self.clean_files = sorted(self.clean_dir.glob("img_*.png"))
        if size is not None:
            self.clean_files = self.clean_files[:size]

        self.add_map: Dict[str, List[Path]] = {}
        self.mul_map: Dict[str, List[Path]] = {}

        for cf in self.clean_files:
            print(f"Processing file: {cf}")
            m = re.match(r"img_(\d+)\.png", cf.name)
            if not m: 
                continue
            cid = m.group(1)
            adds = sorted(self.add_dir.glob(f"img_{cid}_add_*.png"))
            if adds: self.add_map[cid] = adds
            muls = sorted(self.mul_dir.glob(f"img_{cid}_mul_*.png"))
            if not muls:
                muls = sorted(self.mul_dir.glob(f"image_{cid}_mul_*.png"))
            if muls: self.mul_map[cid] = muls

        self.ids = [re.match(r"img_(\d+)\.png", p.name).group(1)
                    for p in self.clean_files
                    if (re.match(r"img_(\d+)\.png", p.name)
                        and (self.add_map.get(re.match(r"img_(\d+)\.png", p.name).group(1))
                             or self.mul_map.get(re.match(r"img_(\d+)\.png", p.name).group(1))))]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_rgb01(path: Path) -> torch.Tensor:
        # tvio.read_image -> uint8 [C,H,W], RGB
        t = tvio.read_image(str(path))  # uint8
        # optional: assert t.shape[-2:] == (256, 256)
        return t.float().div_(255.0)    # [C,H,W] in [0,1]

    def __getitem__(self, idx):
        cid = self.ids[idx]
        clean_path = self.clean_dir / f"img_{cid}.png"

        choices = []
        if cid in self.add_map: choices.append("add")
        if cid in self.mul_map: choices.append("mul")
        noise_type = random.choice(choices)

        if noise_type == "add":
            noisy_path = random.choice(self.add_map[cid])
            noise_label = 0
        else:
            noisy_path = random.choice(self.mul_map[cid])
            noise_label = 1

        clean_t = self._read_rgb01(clean_path)  # [3,H,W] in [0,1]
        noisy_t = self._read_rgb01(noisy_path)  # [3,H,W] in [0,1]
        return noisy_t, clean_t, noise_label

# ---------------------------
# Model (Two-branch)
# ---------------------------
class BranchA_Filters(nn.Module):
    """
    Per spec:
    - split RGB; per channel apply 8 learned 11x11 filters (linear, no activation).
    - Implement as grouped conv: in=3, out=24 (8 per channel), groups=3, k=11, padding=5
    - Output: [B, 24, H, W]  (8 maps per color in order R[0..7], G[0..7], B[0..7])
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=11, padding=5, bias=True, groups=3)
        # Kaiming normal small init is fine for linear filters too
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)  # [B,24,H,W]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, p_drop=0.0):
        super().__init__()
        self.use_proj = (in_ch != out_ch)
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1, bias=not use_bn)
        self.bn1   = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=not use_bn)
        self.bn2   = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.proj  = nn.Conv2d(in_ch, 32, 1) if self.use_proj else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_proj:
            identity = self.proj(identity)
        out = out + identity
        out = self.relu(out)
        return out

class BranchB_Weights(nn.Module):
    """
    Per spec:
    - 3 ResNet blocks, 32 channels, 3x3, BN/Dropout, ReLU, residual.
    - Final 1x1 conv to 8 channels + softmax (per-pixel across channel dim).
    Output: [B, 8, H, W] (weights summing to 1 along channel=8)
    """
    def __init__(self, use_bn=True, p_drop=0.0):
        super().__init__()
        self.block1 = ResBlock(3, 32, use_bn=use_bn, p_drop=p_drop)
        self.block2 = ResBlock(32, 32, use_bn=use_bn, p_drop=p_drop)
        self.block3 = ResBlock(32, 32, use_bn=use_bn, p_drop=p_drop)
        self.to8    = nn.Conv2d(32, 8, kernel_size=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        logits = self.to8(out)                    # [B,8,H,W]
        weights = F.softmax(logits, dim=1)        # channel-wise softmax per pixel
        return weights

class EdgePreservingDenoiser(nn.Module):
    """
    Fusion:
    - Branch A: 24 maps = 8 per color channel (R,G,B).
    - Branch B: 8 weights per pixel.
    - For each color c, take its 8 maps A_c (B,8,H,W), multiply by weights (B,8,H,W), sum over 8 -> (B,1,H,W)
    - Stack 3 colors -> (B,3,H,W)
    """
    def __init__(self, use_bn=True, p_drop=0.0):
        super().__init__()
        self.branchA = BranchA_Filters()
        self.branchB = BranchB_Weights(use_bn=use_bn, p_drop=p_drop)

    def forward(self, x):
        A = self.branchA(x)          # [B,24,H,W]
        W = self.branchB(x)          # [B,8,H,W]

        # split A into R,G,B groups of 8
        A_R = A[:, 0:8, :, :]        # [B,8,H,W]
        A_G = A[:, 8:16, :, :]
        A_B = A[:,16:24, :, :]

        # weighted sum per color
        R = (A_R * W).sum(dim=1, keepdim=True)   # [B,1,H,W]
        G = (A_G * W).sum(dim=1, keepdim=True)
        B = (A_B * W).sum(dim=1, keepdim=True)

        out = torch.cat([R,G,B], dim=1)          # [B,3,H,W]
        return out

# ---------------------------
# Training
# ---------------------------
def train(
    data_root="dataset",
    batch_size=8,
    epochs=5,
    lr=1e-3,
    steps_per_epoch: Optional[int]=None,
    use_bn=True,
    p_drop=0.0,
    num_workers=4,
    resume: Optional[str]=None,
    save_path="denoiser.pth",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ds = NoisyCleanDataset(data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    print(f"Loaded dataset with {len(ds)} samples")
    print(f"Creating model...")
    model = EdgePreservingDenoiser(use_bn=use_bn, p_drop=p_drop).to(device)
    if resume and os.path.isfile(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"Loaded checkpoint: {resume}")
    else:
        print(f"No checkpoint found, starting from scratch")

    print(f"Creating optimizer...")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Starting training...")
    model.train()
    global_step = 0
    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        running_loss, running_psnr = 0.0, 0.0
        for i, (noisy, clean, _) in enumerate(dl):
            print(f"Step {i+1}/{len(dl)}")
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            print(f"Predicting...")
            pred = model(noisy)
            loss = loss_fn(pred, clean)

            print(f"Backpropagating...")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # metrics
            print(f"Calculating metrics...")
            with torch.no_grad():
                batch_psnr = sum(psnr(pred[b], clean[b]) for b in range(pred.size(0))) / pred.size(0)

            print(f"Updating metrics...")
            running_loss += loss.item()
            running_psnr += batch_psnr
            global_step += 1

            print(f"Checking if we should break...")
            if steps_per_epoch and (i+1) >= steps_per_epoch:
                break

            print(f"Printing progress...")
            if (i+1) % 50 == 0:
                print(f"Epoch {ep+1} | step {i+1} | loss {running_loss/(i+1):.4f} | PSNR {running_psnr/(i+1):.2f} dB")

        print(f"Saving model...")
        print(f"==> Epoch {ep+1}/{epochs} | loss {running_loss/max(1,(i+1)):.4f} | PSNR {running_psnr/max(1,(i+1)):.2f} dB")
        torch.save(model.state_dict(), save_path)
        print(f"Saving model...")
        print(f"Saved: {save_path}")

    print("Training done.")

if __name__ == "__main__":
    # Quick test run (adjust as needed)
    print("Training started...")
    train(
        data_root="dataset",
        batch_size=8,
        epochs=5,            # try 1 for smoke test, increase for real training
        lr=1e-3,
        steps_per_epoch=100, # None to use full epoch; 100 is nice for quick check
        use_bn=True,
        p_drop=0.0,
        num_workers=2,
        save_path="edge_denoiser.pth",
    )
    print("Training done.")

