import os
import glob
import cv2
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stitch import stitch_images

def batch_process(data_dir="data", output_dir="outputs"):
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
    all_images = set()
    for ext in image_extensions:
        all_images.update([os.path.abspath(p) for p in glob.glob(os.path.join(data_dir, ext))])

    if not all_images:
        print(f"❌ 在 {data_dir} 目录下没有找到图片！")
        return

    groups = {}
    for img_path in all_images:
        filename = os.path.basename(img_path)
        prefix = filename.split('_')[0] if '_' in filename else os.path.splitext(filename)[0]
        groups.setdefault(prefix, []).append(img_path)

    for prefix, img_paths in groups.items():
        print(f"\n{'='*60}")
        print(f"📁 正在处理组: [{prefix}]，共包含 {len(img_paths)} 张图片")
        
        # 创建对应的保存文件夹
        group_out_dir = os.path.join(output_dir, prefix)
        os.makedirs(group_out_dir, exist_ok=True)
        
        if len(img_paths) == 1:
            print(f"⚠️ 组 {prefix} 只有一张图片，直接复制保存。")
            img = cv2.imread(img_paths[0])
            out_path = os.path.join(group_out_dir, f"{prefix}_panorama.jpg")
            cv2.imwrite(out_path, img)
            continue

        def sort_key(path):
            name = os.path.basename(path)
            try: return int(os.path.splitext(name)[0].split('_')[1])
            except (IndexError, ValueError): return name
                
        img_paths.sort(key=sort_key)
        print(f"📸 按顺序拼接的图片列表: {[os.path.basename(p) for p in img_paths]}")

        try:
            panorama = stitch_images(img_paths, detector_type='sift', blend_mode='weighted')
            
                        # 保存最终全景图
            out_path = os.path.join(group_out_dir, f"{prefix}_panorama.jpg")
            
            # 【修复中文路径保存问题】
            is_success, im_buf_arr = cv2.imencode(".jpg", panorama)
            if is_success:
                im_buf_arr.tofile(out_path)
                print(f"✅ 成功: 拼接结果已保存至 {out_path}")
            else:
                print(f"❌ 图像编码失败，无法保存: {out_path}")
            
        except Exception as e:
            print(f"❌ 失败: 组 {prefix} 拼接出错。原因: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_path = os.path.join(base_dir, "data")
    output_path = os.path.join(base_dir, "outputs")
    
    batch_process(data_dir=data_path, output_dir=output_path)