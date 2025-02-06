from head_segmentation import segmentation_pipeline as seg_pipeline
import numpy as np 
import cv2
import torch
from PIL import Image
import argparse
import os


def segment(image):
    """
    Segments the input image, turning all non-black pixels in the segmented output to white.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Output image with non-black pixels turned white.
    """
    segmented_image = image 

    # Create a mask for non-black pixels
    non_black_mask = (segmented_image[:, :, 0] != 0) | \
                     (segmented_image[:, :, 1] != 0) | \
                     (segmented_image[:, :, 2] != 0)

    # Turn all non-black pixels to white
    segmented_image[non_black_mask] = [255, 255, 255]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

    return cleaned_image



def process_mask(pil_image: Image.Image) -> Image.Image:
    """
    Process a mask to fill holes, remove small black regions, and smooth edges.
    
    Args:
        pil_image (PIL.Image.Image): Input mask as a PIL image.
        
    Returns:
        PIL.Image.Image: Processed mask as a PIL image.
    """

    mask = np.array(pil_image.convert("L")) 

    # Step 1: Fill holes using flood fill
    filled_mask = mask.copy()
    h, w = mask.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)  # +2 for floodFill boundary

    cv2.floodFill(filled_mask, flood_fill_mask, (0, 0), 255)
    filled_mask = cv2.bitwise_or(mask, cv2.bitwise_not(filled_mask))

    # Step 2: Remove small black regions using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: (Optional) Smooth the edges
    smoothed_mask = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)

    processed_pil_image = Image.fromarray(smoothed_mask)

    return processed_pil_image




def annotate(img: Image.Image):
    """ Generates annotation image (segmentation) for the given image. 
    
    Arguments:
        image: image to segment. 
    Returns:
        mask_image: annotated segment(s) as a binary image or None if no people in img.
    """
    # Convert image to RGB
    img = img.convert("RGB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)
    
    img_array = np.array(img)
    
    segmentation_map = segmentation_pipeline(img_array)
    final = img_array * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
    
    masked_image = segment(final)
    mask_image = Image.fromarray(masked_image)
    
    mask_image = mask_image.convert("L")  # Convert mask to grayscale
    mask_data = np.array(mask_image)
    binary_mask = np.where(mask_data > 127, 255, 0).astype(np.uint8)
    final_mask_image = Image.fromarray(binary_mask)
    smoothed_mask_image = process_mask(final_mask_image)

    return smoothed_mask_image


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./masks')
    args = parser.parse_args()
    
    print("Segmentation has started!")
    image = Image.open(args.im_path).convert("RGBA")
    
    mask = annotate(image)
    
    im_name = os.path.splitext(os.path.basename(args.im_path))[0]
    save_path = os.path.join(args.save_folder, f"{im_name}_mask.png")
    
    mask.save(save_path)
    print(f"Segmentation has finished and saved to {save_path}!")
    
    


