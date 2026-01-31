
import os
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def decode_jp2_to_png(input_path, output_path):
    """
    Decode a JP2 image and save it as PNG.
    
    Args:
        input_path: Path to the input JP2 file
        output_path: Path to save the output PNG file
    """
    try:
        # Try using glymur first (best for JPEG 2000)
        try:
            import glymur
            jp2_image = glymur.Jp2k(input_path)
            img_array = jp2_image[:]
            
            # Normalize the image data for visualization
            if img_array.dtype != np.uint8:
                # Normalize to 0-255 range
                img_min = np.nanmin(img_array)
                img_max = np.nanmax(img_array)
                if img_max > img_min:
                    img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_normalized = np.zeros_like(img_array, dtype=np.uint8)
            else:
                img_normalized = img_array
            
            # Convert to PIL Image and save as PNG
            if len(img_normalized.shape) == 2:
                # Grayscale image
                img = Image.fromarray(img_normalized, mode='L')
            elif len(img_normalized.shape) == 3:
                if img_normalized.shape[2] == 3:
                    # RGB image
                    img = Image.fromarray(img_normalized, mode='RGB')
                elif img_normalized.shape[2] == 4:
                    # RGBA image
                    img = Image.fromarray(img_normalized, mode='RGBA')
                else:
                    # Take first channel for multi-band images
                    img = Image.fromarray(img_normalized[:, :, 0], mode='L')
            else:
                raise ValueError(f"Unexpected image shape: {img_normalized.shape}")
            
            img.save(output_path, 'PNG')
            print(f"✓ Decoded: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            return True
            
        except ImportError:
            # Fallback to PIL/Pillow
            print("Note: glymur not available, using PIL/Pillow (may be slower)")
            img = Image.open(input_path)
            
            # Convert to numpy array for normalization
            img_array = np.array(img)
            
            # Normalize if needed
            if img_array.dtype != np.uint8:
                img_min = np.nanmin(img_array)
                img_max = np.nanmax(img_array)
                if img_max > img_min:
                    img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_normalized = np.zeros_like(img_array, dtype=np.uint8)
                img = Image.fromarray(img_normalized)
            
            img.save(output_path, 'PNG')
            print(f"✓ Decoded: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            return True
            
    except Exception as e:
        print(f"✗ Error decoding {os.path.basename(input_path)}: {str(e)}")
        return False


def main():
    """Main function to process all JP2 images in R20m directory."""
    
    # Define paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'data' / 'IMG_DATA' / 'R20m'
    output_dir = script_dir / 'data' / 'decoded' / 'R20m'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all JP2 files
    jp2_files = sorted(input_dir.glob('*.jp2'))
    
    if not jp2_files:
        print(f"No JP2 files found in {input_dir}")
        return
    
    print(f"\nFound {len(jp2_files)} JP2 files to process\n")
    
    # Process each JP2 file
    successful = 0
    failed = 0
    
    for jp2_file in jp2_files:
        # Create output filename (replace .jp2 with .png)
        output_filename = jp2_file.stem + '.png'
        output_path = output_dir / output_filename
        
        # Decode and save
        if decode_jp2_to_png(str(jp2_file), str(output_path)):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully decoded: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Output location: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

