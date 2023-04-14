from PIL import Image
import numpy as np
import os

def add_noise(image_path, noise_type='gaussian', mean=0, std=25, s_vs_p=0.5, amount=0.05):
    """
    给指定的PNG图像添加噪声并保存为新的PNG文件。

    Args:
        image_path (str): 要添加噪声的PNG图像路径。
        noise_type (str): 噪声类型，可以是'gaussian'、'salt_pepper'或'poisson'中的一种，默认为'gaussian'。
        mean (float): 高斯噪声的均值，默认为0。
        std (float): 高斯噪声的标准差，默认为25。
        s_vs_p (float): 椒盐噪声中椒盐的比例，默认为0.5。
        amount (float): 椒盐噪声的噪声强度，默认为0.05。

    Returns:
        None
    """
    # 加载PNG图像
    image = Image.open(image_path)

    # 转换图像为numpy数组
    image_np = np.array(image)

    # 添加指定类型的噪声
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, std, image_np.shape)
        noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noisy_image = add_salt_pepper_noise(image_np, s_vs_p, amount)
    elif noise_type == 'poisson':
        noisy_image = add_poisson_noise(image_np)
    else:
        raise ValueError(f"Unsupported noise type '{noise_type}', should be 'gaussian', 'salt_pepper' or 'poisson'.")
    if not os.path.exists(noise_type):
        os.makedirs(noise_type)
    # 保存噪声图像为PNG格式
    noisy_image =Image.fromarray(noisy_image)
    noisy_image.save(f"{noise_type}{image_path}")
    return noisy_image


def add_salt_pepper_noise(image, s_vs_p, amount):
    """
    给指定的图像添加椒盐噪声。

    Args:
        image (np.ndarray): 要添加噪声的图像。
        s_vs_p (float): 椒盐噪声中椒盐的比例。
        amount (float): 椒盐噪声的噪声强度。

    Returns:
        np.ndarray: 添加了椒盐噪声的图像。
    """
    row, col, ch = image.shape
    num_salt = int(np.ceil(amount * row * col * s_vs_p))
    num_pepper = int(np.ceil(amount * row * col * (1.0 - s_vs_p)))

    # 添加椒盐噪声
    noisy_image = np.copy(image)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords_salt[0], coords_salt[1], :] = 255
    noisy_image[coords_pepper[0], coords_pepper[1], :] = 0
    return noisy_image

def add_poisson_noise(image):
    """
    给指定的图像添加泊松噪声。

    Args:
        image (np.ndarray): 要添加噪声的图像。

    Returns:
        np.ndarray: 添加了泊松噪声的图像。
    """
    # 添加泊松噪声
    noisy_image = np.copy(image)
    for i in range(noisy_image.shape[2]):
        noisy_image[..., i] = np.random.poisson(noisy_image[..., i])
    return noisy_image


if __name__=="__main__":
    image_path = "./test2.png"
    add_noise(image_path, noise_type='gaussian', mean=0, std=25)
    add_noise(image_path, noise_type='salt_pepper', s_vs_p=0.5, amount=0.05)
    add_noise(image_path, noise_type='poisson')
    
    #nature, weapon, long hair, black hair, reflection, forest, sword, bamboo, solo, bamboo forest, water, katana, outdoors, lake, scenery, pants, from behind, 1girl, 1boy, male focus
    #poisson nature, bamboo, long hair, black hair, solo, reflection, forest, bamboo forest, pants, water, outdoors, 1girl, scenery, shirt, white shirt, walking, from behind, 1boy
    #gaussian nature, forest, solo, bamboo, weapon, black hair, long hair, sword, reflection, bamboo forest, 1girl, pants, outdoors, scenery, katana, from behind, water, traditional media, lake, shirt
    #salt solo, weapon, nature, black hair, long hair, bamboo, forest, reflection, sword, bamboo forest, outdoors, pants, traditional media, from behind, scenery, japanese clothes, 1girl, 1boy, water, male focus