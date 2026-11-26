import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def overlap_ssim(img1, img2, H):
    """计算 img2 变换到 img1 坐标系后，重叠区域的 SSIM。"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img2, H, (w1, h1))

    mask1 = np.ones((h1, w1), dtype=np.uint8) * 255
    mask2 = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8) * 255, H, (w1, h1))
    overlap_mask = cv2.bitwise_and(mask1, mask2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray_warped, full=True, data_range=255, mask=overlap_mask > 0)
    return score