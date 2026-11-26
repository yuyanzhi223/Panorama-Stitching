import cv2
import numpy as np
import argparse
import os

def cv2_imread_unicode(image_path, flags=cv2.IMREAD_COLOR):
    """兼容中文路径"""
    image_array = np.fromfile(image_path, dtype=np.uint8)
    return cv2.imdecode(image_array, flags)

def stitch_with_opencv_stitcher(image_paths):
    """
    【新增核心优化】使用 OpenCV 官方的高级 Stitcher 类进行拼接。
    它内置了柱面/球面投影、特征匹配、曝光补偿和接缝查找，非常适合处理 boat 这种广角场景。
    """
    print("🌟 尝试使用 OpenCV 官方高级 Stitcher 接口（支持柱面投影，消除波浪畸变）...")
    images = []
    for p in image_paths:
        img = cv2_imread_unicode(p)
        if img is not None:
            images.append(img)
            
    # 尝试不同的模式，PANORAMA 适合多图长条，SCANS 适合平面文档
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched_img = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            print("✅ OpenCV Stitcher 拼接成功！")
            return stitched_img
        else:
            print(f"⚠️ OpenCV 官方接口失败 (状态码: {status})，退回手动拼接。")
    except Exception as e:
        print(f"⚠️ OpenCV 官方接口异常: {e}，退回手动拼接。")
    
    return None

# ----------------- 以下为原有的手动拼接方法（作为兜底） -----------------

def detect_and_match(img1, img2, detector_type='sift'):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if detector_type == 'orb':
        detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)
        norm_type = cv2.NORM_HAMMING
    elif detector_type == 'akaze':
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    elif detector_type == 'sift':
        detector = cv2.SIFT_create(nfeatures=4000)
        norm_type = cv2.NORM_L2 
    else:
        raise ValueError("detector_type must be 'orb', 'akaze' or 'sift'")

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Not enough descriptors found in one of the images.")

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for match in knn_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 10:
        raise RuntimeError(f"Too few good matches: {len(good)}. 放弃拼接此图。")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return pts1, pts2, good, kp1, kp2

def compute_homography(pts1, pts2):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H, mask

def blend_direct(warped, base, mask_warped):
    result = base.copy()
    result[mask_warped > 0] = warped[mask_warped > 0]
    return result

def blend_weighted(warped, base, mask_warped, alpha=0.5):
    result = base.copy()
    overlap = (mask_warped > 0) & (base.sum(axis=2) > 0)
    result[overlap] = cv2.addWeighted(base[overlap], 1 - alpha, warped[overlap], alpha, 0)
    non_overlap_warped = (mask_warped > 0) & (base.sum(axis=2) == 0)
    result[non_overlap_warped] = warped[non_overlap_warped]
    return result

def warp_and_blend(base_img, next_img, H, blend_mode='weighted'):
    h_base, w_base = base_img.shape[:2]
    h_next, w_next = next_img.shape[:2]

    corners_next = np.float32([[0, 0], [0, h_next], [w_next, h_next], [w_next, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_next, H)
    all_corners = np.concatenate([np.float32([[0, 0], [0, h_base], [w_base, h_base], [w_base, 0]]).reshape(-1, 1, 2),
                                  warped_corners], axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w, canvas_h = x_max - x_min, y_max - y_min
    
    if canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError("计算出的画布尺寸异常。")

    base_canvas = cv2.warpPerspective(base_img, translation, (canvas_w, canvas_h))
    warped_canvas = cv2.warpPerspective(next_img, translation @ H, (canvas_w, canvas_h))
    mask_warped = cv2.warpPerspective(np.ones((h_next, w_next), dtype=np.uint8) * 255, translation @ H, (canvas_w, canvas_h))

    if blend_mode == 'direct':
        result = blend_direct(warped_canvas, base_canvas, mask_warped)
    elif blend_mode == 'weighted':
        result = blend_weighted(warped_canvas, base_canvas, mask_warped, alpha=0.5)
    return result

def stitch_images_manual(image_paths, detector_type='sift', blend_mode='weighted'):
    images = []
    MAX_WIDTH = 1600 
    for p in image_paths:
        img = cv2_imread_unicode(p)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {p}")
        if img.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        images.append(img)

    mid_idx = len(images) // 2
    panorama = images[mid_idx]
    print(f"Using Image {mid_idx} as the base canvas.")
    
    try:
        for i in range(mid_idx + 1, len(images)):
            print(f"Stitching right image {i}...")
            pts1, pts2, good, kp1, kp2 = detect_and_match(panorama, images[i], detector_type)
            H, mask = compute_homography(pts2, pts1) 
            panorama = warp_and_blend(panorama, images[i], H, blend_mode)
    except Exception as e:
        print(f"⚠️ 向右拼接中断: {e}")

    try:
        for i in range(mid_idx - 1, -1, -1):
            print(f"Stitching left image {i}...")
            pts1, pts2, good, kp1, kp2 = detect_and_match(panorama, images[i], detector_type)
            H, mask = compute_homography(pts2, pts1)
            panorama = warp_and_blend(panorama, images[i], H, blend_mode)
    except Exception as e:
        print(f"⚠️ 向左拼接中断: {e}")

    # 裁剪黑边
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        panorama = panorama[y:y+h, x:x+w]

    return panorama

# 【对外主接口】优先调用官方，失败则调用手动
def stitch_images(image_paths, detector_type='sift', blend_mode='weighted'):
    result = stitch_with_opencv_stitcher(image_paths)
    if result is not None:
        return result
    else:
        print("🔄 使用手动单应性矩阵进行兜底拼接...")
        return stitch_images_manual(image_paths, detector_type, blend_mode)