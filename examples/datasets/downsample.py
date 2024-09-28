'''
Created by Han Xu
email:736946693@qq.com
'''
import os
import cv2
import numpy
from PIL import Image
def downsample_image(image_path, factor, output_folder):
    with Image.open(image_path) as img:
        new_width = img.width // factor
        new_height = img.height // factor
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        base_name = os.path.basename(image_path)
        new_image_path = os.path.join(output_folder, base_name)
        img_resized.save(new_image_path)
        print(f"Processed {base_name}")


def downsample_batch(input_folder, factor):
    output_folder = f"{input_folder}_{factor}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(input_folder, file_name)
            downsample_image(file_path, factor, output_folder)

from PIL import Image
import os
import numpy as np
import cv2

def crop_image(image_path, output_folder, target_width=200, target_height=200):
    # 使用Pillow加载图片，自动处理各种格式
    image = Image.open(image_path).convert("RGBA")  # 保证图片格式为RGBA，便于处理透明背景
    image_np = np.array(image)  # 转换为numpy数组

    # 提取Alpha通道
    alpha_channel = image_np[:, :, 3]
    _, binary = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"图像 {image_path} 没有找到有效的物体轮廓，跳过处理。")
        return [], []

    # 获取最大轮廓的边界框 (ROI)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    roi = image_np[y:y+h, x:x+w]

    # 计算分块数量和间隙
    num_cols = w // target_width
    num_rows = h // target_height

    extra_width = w - num_cols * target_width
    extra_height = h - num_rows * target_height

    gap_x = extra_width / (num_cols + 1)
    gap_y = extra_height / (num_rows + 1)

    patches = []
    percentages = []

    for row in range(num_rows):
        for col in range(num_cols):
            start_x = int(col * target_width + (col + 1) * gap_x)
            start_y = int(row * target_height + (row + 1) * gap_y)
            end_x = start_x + target_width
            end_y = start_y + target_height

            patch = roi[start_y:end_y, start_x:end_x]
            patches.append(patch)

            # 计算非透明像素的百分比
            non_transparent_percentage = calculate_non_transparent_percentage(patch)
            percentages.append(non_transparent_percentage)

    return patches, percentages

def calculate_non_transparent_percentage(patch):
    # 提取Alpha通道
    if patch.shape[2] == 4:
        alpha_channel = patch[:, :, 3]  # Alpha通道
        total_pixels = alpha_channel.size
        non_transparent_pixels = np.count_nonzero(alpha_channel)  # 非透明的像素个数
        return non_transparent_pixels / total_pixels * 100  # 返回非透明像素的百分比
    else:
        return 100  # 如果没有Alpha通道，默认认为全是非透明

# 处理文件夹中的所有图片
def crop_batch(input_folder, output_folder, target_width=200, target_height=200):
    # 支持多种图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    all_patches = []
    all_percentages = []
    patch_filenames = []

    # 遍历所有图片，收集图块和非透明像素占比
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        base_name = os.path.splitext(image_file)[0]

        patches, percentages = crop_image(image_path, output_folder, target_width, target_height)
        all_patches.extend(patches)
        all_percentages.extend(percentages)

        for i in range(len(patches)):
            patch_filenames.append(f'{base_name}_{i + 1}.png')

    if not all_patches:
        print("没有找到有效的图块进行处理。")
        return

    # 计算总体的均值和方差
    miu = np.mean(all_percentages)
    sigma = np.std(all_percentages)
    threshold = miu + 0.5 * sigma

    print(f"总体均值: {miu:.2f}%, 方差: {sigma:.2f}%")
    print(f"筛选阈值: {threshold:.2f}%")

    # 保存符合条件的图块
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for file_name in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file_name))

    saved_patch_count = 0
    for i, percentage in enumerate(all_percentages):
        if percentage >= threshold:
            output_path = os.path.join(output_folder, patch_filenames[i])
            # 使用Pillow保存图像，确保保持透明背景
            patch_image = Image.fromarray(all_patches[i])
            patch_image.save(output_path, format='PNG')
            saved_patch_count += 1

    print(f"共保存了 {saved_patch_count} 个符合条件的图块到 {output_folder}。")


if __name__ == "__main__":
    # downsample_batch(rf"{os.path.dirname(__file__)}/lego_crop/train", 2)

    input_folder = rf"{os.path.dirname(__file__)}/vasedeck/images"
    output_folder = rf"{os.path.dirname(__file__)}/vasedeck_crop/images"

    # 处理所有图片
    crop_batch(input_folder, output_folder, target_width=200, target_height=200)

    print("所有图片处理完成。")