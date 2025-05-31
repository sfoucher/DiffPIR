import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_random_mask(image_size=256, mask_type='granular', density=0.1, output_path='random_mask.png'):
    """
    生成一个 256x256 的随机mask，支持多种类型。

    参数：
        image_size (int): 图像的尺寸（默认 256x256）
        mask_type (str): mask类型，可选 'granular'（颗粒状）、'spots'（斑点）、'stripes'（条纹）、'blocks'（随机块）、'centered'（居中）
        density (float): 噪声密度（0到1之间），控制mask中损坏区域的比例
        output_path (str): 保存mask的路径
    """
    # 创建一个 256x256 的空白图像，初始值为255（白色）
    mask = np.ones((image_size, image_size), dtype=np.uint8) * 255

    if mask_type == 'centered':
        # 居中mask（原逻辑）
        mask_size = 64
        center = image_size // 2
        half_mask = mask_size // 2
        top_left = center - half_mask
        bottom_right = center + half_mask
        mask[top_left:bottom_right, top_left:bottom_right] = 0

    elif mask_type == 'granular':
        # 颗粒状噪声（类似示例图片）
        np.random.seed(42)  # 固定随机种子以便复现
        noise = np.random.binomial(1, density, size=(image_size, image_size))  # 二值噪声
        mask[noise == 1] = 0  # 噪声点设为黑色

    elif mask_type == 'spots':
        # 斑点噪声（随机圆形斑点）
        np.random.seed(42)
        num_spots = int(density * 100)  # 根据密度确定斑点数量
        for _ in range(num_spots):
            # 随机中心点
            center_x = np.random.randint(0, image_size)
            center_y = np.random.randint(0, image_size)
            # 随机半径
            radius = np.random.randint(5, 20)
            # 绘制圆形斑点
            for x in range(image_size):
                for y in range(image_size):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                        mask[x, y] = 0

    elif mask_type == 'stripes':
        # 条纹噪声（随机方向条纹）
        np.random.seed(42)
        num_stripes = int(density * 20)  # 根据密度确定条纹数量
        for _ in range(num_stripes):
            # 随机选择条纹方向（水平或垂直）
            direction = np.random.choice(['horizontal', 'vertical'])
            # 随机选择条纹位置和宽度
            if direction == 'horizontal':
                y = np.random.randint(0, image_size)
                thickness = np.random.randint(2, 10)
                mask[max(0, y - thickness):min(image_size, y + thickness), :] = 0
            else:
                x = np.random.randint(0, image_size)
                thickness = np.random.randint(2, 10)
                mask[:, max(0, x - thickness):min(image_size, x + thickness)] = 0

    elif mask_type == 'blocks':
        # 随机块噪声（随机矩形块）
        np.random.seed(42)
        num_blocks = int(density * 20)  # 根据密度确定块数量
        for _ in range(num_blocks):
            # 随机块大小
            block_width = np.random.randint(10, 50)
            block_height = np.random.randint(10, 50)
            # 随机块位置
            top_left_x = np.random.randint(0, image_size - block_width)
            top_left_y = np.random.randint(0, image_size - block_height)
            # 绘制矩形块
            mask[top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 0

    else:
        raise ValueError(f"不支持的mask类型: {mask_type}")

    # 保存mask
    mask_image = Image.fromarray(mask, mode='L')
    mask_image.save(output_path)
    print(f"Mask 已保存到 {output_path}")

    # 可视化
    plt.imshow(mask, cmap='gray')
    plt.title(f'{mask_type.capitalize()} Mask (Density: {density})')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 生成不同类型的mask
    mask_types = ['granular', 'spots', 'stripes', 'blocks', 'centered']
    for mask_type in mask_types:
        output_path = f'masks/{mask_type}_mask.png'
        create_random_mask(image_size=256, mask_type=mask_type, output_path=output_path)