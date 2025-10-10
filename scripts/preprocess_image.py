import numpy as np
from tensorflow.keras.preprocessing import image


def load_image(
    img_path: str,
    target_size: tuple[int, int]
):
    # Load ảnh và resize
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    # Convert sang numpy array
    img_array = image.img_to_array(img)
    # Expand thành batch (shape: (1, h, w, c))
    img_array = np.expand_dims(img_array, axis=0)
    # Chuẩn hóa về [0,1]
    img_array = img_array / 255.0
    return img_array