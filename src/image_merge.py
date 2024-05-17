import numpy as np
import cv2


class ImageProcessor:
    def __init__(self, lambda_r, lambda_g, lambda_b, alpha=255, brightness_ratio=1.0):
        self.lambda_r = lambda_r
        self.lambda_g = lambda_g
        self.lambda_b = lambda_b
        self.alpha = alpha
        self.brightness_ratio = brightness_ratio

    def color_translation(self, power):
        """Translate power P to RGB values based on the QEs."""
        y_r = self.relu(power * self.lambda_r)
        y_g = self.relu(power * self.lambda_g)
        y_b = self.relu(power * self.lambda_b)
        return np.stack((y_r, y_g, y_b), axis=-1)

    def relu(self, x):
        """ReLU-like function to handle pixel saturation."""
        return np.minimum(x, self.alpha)

    def apply_laser_perturbation(self, input_image, laser_power):
        """Apply laser perturbation and handle pixel saturation."""
        laser_signal = self.color_translation(laser_power)
        perturbed_image = self.relu(input_image + laser_signal)
        return perturbed_image

    def adjust_brightness(self, synthetic_image, p_o, p_a):
        """Adjust brightness of the synthetic image based on real-world camera exposure."""
        r_b = p_o / (p_o + p_a)
        return synthetic_image * r_b

    def process_image(self, input_image, laser_power, p_o, p_a):
        """Process the image by applying laser perturbation and adjusting brightness."""
        # Apply laser perturbation
        perturbed_image = self.apply_laser_perturbation(input_image, laser_power)
        # Adjust brightness
        final_image = self.adjust_brightness(perturbed_image, p_o, p_a)
        return final_image

    def laser_shape_adjust(self, img_laser, img_face):
        """Adjust the shape of the laser image to match the face image dimensions."""
        height_laser, width_laser, _ = img_laser.shape
        height_face, width_face, _ = img_face.shape
        laser_cropped = img_laser[height_laser // 2 - width_face // 2:height_laser // 2 + width_face // 2,
                        width_laser // 2 - width_face // 2:width_laser // 2 + width_face // 2]
        return laser_cropped

    def simple_add(self, base_img, light_pattern):
        """Add the light pattern to the base image with given alpha and beta parameters."""
        return cv2.addWeighted(base_img, self.alpha, light_pattern, self.beta, 0)

    def coordinate(self, center_x, center_y, base_img, added_img):
        """Overlay images at specific coordinates."""
        rows_base, cols_base, channel = base_img.shape
        rows_laser, cols_laser, channel_laser = added_img.shape
        new_image = np.zeros((rows_base + rows_laser, cols_base + cols_laser, channel), dtype=base_img.dtype)
        new_image2 = np.zeros((rows_base + rows_laser, cols_base + cols_laser, channel), dtype=base_img.dtype)

        new_image[rows_laser // 2: rows_base + rows_laser // 2, cols_laser // 2: cols_base + cols_laser // 2] = base_img
        new_image2[center_y: center_y + rows_laser, center_x: center_x + cols_laser] = added_img

        base_img_roi = cv2.add(new_image, new_image2)
        base_img_roi[base_img_roi > 255] = 255

        laser_face = np.zeros(base_img.shape, dtype=base_img.dtype)
        laser_face[:, :] = base_img_roi[rows_laser // 2: rows_base + rows_laser // 2,
                           cols_laser // 2: cols_base + cols_laser // 2]

        return laser_face


if __name__ == '__main__':

    lambda_r = 0.9
    lambda_g = 0.8
    lambda_b = 0.7
    image_processor = ImageProcessor(lambda_r, lambda_g, lambda_b, alpha=255, brightness_ratio=1.0)

    # Example image loading (you need to load images using OpenCV or any other library)
    img_face = cv2.imread('path_to_face_image')
    img_laser = cv2.imread('path_to_laser_image')

    # Define the power of the laser and input image
    laser_power = 1.0  # Example value
    p_o = 100  # Example value for power of input image
    p_a = 50  # Example value for laser intensity

    # Process the image
    final_image = image_processor.process_image(img_face, laser_power, p_o, p_a)

    # Save or display the final image as needed
    cv2.imwrite('path_to_save_final_image', final_image)
