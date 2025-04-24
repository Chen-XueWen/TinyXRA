import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate explanation plots.')
parser.add_argument("--test_year", default="2024", type=int)
parser.add_argument("--max_docs", default=100, type=int)
args = parser.parse_args()

risk_metrics = ['std', 'skew', 'kurt', 'sortino']
output_dir = f"comparison_heatmaps/{args.test_year}"
os.makedirs(output_dir, exist_ok=True)

max_docs = args.max_docs

for i in tqdm(range(0, max_docs + 1)):
    image_paths = [f"attention_heatmaps/{rm}/{args.test_year}/Doc_{i}_top5_word_attention.png" for rm in risk_metrics]

    if all(os.path.exists(p) for p in image_paths):
        imgs = [cv2.imread(p) for p in image_paths]

        top = np.hstack(imgs[:2])
        bottom = np.hstack(imgs[2:])
        grid = np.vstack([top, bottom])

        output_path = os.path.join(output_dir, f'Doc_{i}_comparison.png')
        cv2.imwrite(output_path, grid)
