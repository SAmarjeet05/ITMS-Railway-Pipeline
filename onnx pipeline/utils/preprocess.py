import cv2
import numpy as np

def preprocess_image(image_path, img_size=640, return_shape=False, normalize=True):
    """
    Reads an image, applies letterbox resize to keep aspect ratio,
    normalizes, and converts to tensor for ONNX.
    Returns the preprocessed image and original image shape info for scaling boxes.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h_orig, w_orig = img.shape[:2]

    # Compute scale and new size for letterbox
    scale = min(img_size / w_orig, img_size / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)

    # Resize image with aspect ratio
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding to center the image
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    pad_left = pad_x
    pad_right = img_size - new_w - pad_x
    pad_top = pad_y
    pad_bottom = img_size - new_h - pad_y

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    if normalize:
        img_norm = img_rgb.astype(np.float32) / 255.0
    else:
        img_norm = img_rgb.astype(np.float32)

    # Convert to (1,3,H,W) tensor
    input_tensor = np.transpose(img_norm, (2, 0, 1))[None, ...]

    if return_shape:
        # Return all info needed for box rescaling
        return input_tensor, (h_orig, w_orig, scale, pad_left, pad_top)
    return input_tensor
