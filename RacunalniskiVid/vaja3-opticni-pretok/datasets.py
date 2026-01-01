import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from IO import readImage, readFlow


class FlyingChairsDataset(Dataset):
    def __init__(self, indices, data_dir="./data"):
        self.indices = indices
        self.data_dir = data_dir

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        img1_path = os.path.join(self.data_dir, f"{idx}_img1.ppm")
        img2_path = os.path.join(self.data_dir, f"{idx}_img2.ppm")
        flow_path = os.path.join(self.data_dir, f"{idx}_flow.flo")

        img1 = readImage(img1_path).astype(np.float32) / 255.0
        img2 = readImage(img2_path).astype(np.float32) / 255.0
        flow = readFlow(flow_path).astype(np.float32)

        # Convert from HWC to CHW format for PyTorch
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)

        return img1, img2, flow


class TestOpticalFlowDataset(Dataset):
    def __init__(self, data_dir, indices=None, img_ext=".png", skip_corrupted=True):
        self.data_dir = Path(data_dir)
        self.img_ext = img_ext
        self.skip_corrupted = skip_corrupted

        if indices is None:
            # Discover all *_img1 files and extract indices
            all_indices = sorted(
                [p.name.split("_")[0] for p in self.data_dir.glob(f"*_img1{img_ext}")]
            )
        else:
            all_indices = list(indices)

        # Validate files if skip_corrupted is enabled
        if skip_corrupted:
            self.indices = []
            corrupted_count = 0
            for idx in all_indices:
                try:
                    # Try to load the flow file to check if it's valid
                    flow_path = self.data_dir / f"{idx}_flow.flo"
                    _ = readFlow(str(flow_path))
                    self.indices.append(idx)
                except Exception as e:
                    corrupted_count += 1
                    print(f"Skipping corrupted sample {idx}: {str(e)}")
            
            if corrupted_count > 0:
                print(f"Skipped {corrupted_count} corrupted samples")
        else:
            self.indices = all_indices

        print(f"TestOpticalFlowDataset: {len(self.indices)} valid samples found in {self.data_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        img1_path = self.data_dir / f"{idx}_img1{self.img_ext}"
        img2_path = self.data_dir / f"{idx}_img2{self.img_ext}"
        flow_path = self.data_dir / f"{idx}_flow.flo"

        img1 = readImage(str(img1_path)).astype(np.float32) / 255.0
        img2 = readImage(str(img2_path)).astype(np.float32) / 255.0
        flow = readFlow(str(flow_path)).astype(np.float32)

        # Convert from HWC to CHW format for PyTorch
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)

        return img1, img2, flow

