import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def create_title_image(text, width, height=100):
    """
    Create a simple title image with given text, centered.
    """
    title_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(title_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    return title_img

def crop_whitespace_partial(image, tol=10, pad_ratio=0.5):
    """
    Crops some of the white space around an image.
    - tol: tolerance for detecting white (default is near-white)
    - pad_ratio: how much of the white space to retain (0.0 = no extra, 1.0 = full original)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < (255 - tol)

    if not np.any(mask):
        return image  # Entire image is white

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Compute original dimensions
    height, width = image.shape[:2]

    # Add back part of the white space
    y_pad = int(((y0) + (height - y1)) * pad_ratio / 2)
    x_pad = int(((x0) + (width - x1)) * pad_ratio / 2)

    y0 = max(0, y0 - y_pad)
    y1 = min(height, y1 + y_pad)
    x0 = max(0, x0 - x_pad)
    x1 = min(width, x1 + x_pad)

    return image[y0:y1, x0:x1]


risk_titles = {
    'std': 'Standard Deviation',
    'skew': 'Skewness',
    'kurt': 'Kurtosis',
    'sortino': 'Sortino Ratio',
}

parser = argparse.ArgumentParser(description='Generate explanation plots.')
parser.add_argument("--test_year", default="2024", type=int)
args = parser.parse_args()

risk_metrics = ['std', 'skew', 'kurt', 'sortino']
output_dir = f"comparison_clouds"
os.makedirs(output_dir, exist_ok=True)

combined_blocks = []

for risk in risk_metrics:
    image_paths = [f"attention_heatmaps/{risk}/{args.test_year}/cloud_label_{i}.png" for i in range(0, 3)]

    if all(os.path.exists(p) for p in image_paths):
        imgs = [crop_whitespace_partial(cv2.imread(p), pad_ratio=0.2) for p in image_paths]

        # Optionally resize to the same height for nicer alignment
        min_height = min(img.shape[0] for img in imgs)
        resized = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in imgs]

        grid = np.hstack(resized)

        # Add title above this row
        title_img = create_title_image(risk_titles[risk], grid.shape[1])
        combined = np.vstack([title_img, grid])
        combined_blocks.append(combined)


# Stack all risk blocks vertically
if combined_blocks:

    # Make all blocks same width for vertical stacking
    max_width = max(block.shape[1] for block in combined_blocks)

    for i in range(len(combined_blocks)):
        h, w = combined_blocks[i].shape[:2]
        if w < max_width:
            pad_width = max_width - w
            # Pad equally on left and right
            left = pad_width // 2
            right = pad_width - left
            combined_blocks[i] = cv2.copyMakeBorder(combined_blocks[i], 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    final_image = np.vstack(combined_blocks)
    final_path = os.path.join(output_dir, f'all_risks_combined_{args.test_year}.png')
    cv2.imwrite(final_path, final_image)