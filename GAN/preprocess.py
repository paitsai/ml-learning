import os
from PIL import Image
from tqdm import tqdm  # 导入 tqdm 库

# 设置目标尺寸
target_size = (512, 512)

# 获取当前目录下的所有 jpg 文件
current_directory = os.getcwd()
jpg_files = [f for f in os.listdir(current_directory) if f.endswith('.jpg')]

# 在处理文件时显示进度条
for jpg_file in tqdm(jpg_files, desc="处理图像", unit="文件"):
    # 打开图像
    img_path = os.path.join(current_directory, jpg_file)
    with Image.open(img_path) as img:
        # 获取原始尺寸
        original_size = img.size
        
        if original_size < target_size:
            # 如果图像小于目标尺寸，则填充
            new_img = Image.new('RGB', target_size, (255, 255, 255))  # 创建白色背景
            new_img.paste(img, ((target_size[0] - original_size[0]) // 2,
                                 (target_size[1] - original_size[1]) // 2))
        else:
            # 如果图像大于等于目标尺寸，则压缩
            new_img = img.resize(target_size, Image.LANCZOS)

        # 保存处理后的图像
        new_img.save(os.path.join(current_directory, f'processed_{jpg_file}'))

print("所有 JPG 图像已处理完成。")