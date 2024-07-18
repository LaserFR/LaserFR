import cv2
import numpy as np
import os


class LaserFaceMerger:
    def __init__(self, alpha=255, laser_intensity=1.0):
        self.alpha = alpha
        self.laser_intensity = laser_intensity

    @staticmethod
    def relu_like(x, alpha=255):
        return np.minimum(x, alpha)

    def merge_laser_face(self, face_image, laser_image, center_coords):
        """Merge the laser image onto the face image at the specified center coordinates"""
        face_h, face_w = face_image.shape[:2]
        laser_h, laser_w = laser_image.shape[:2]
        center_x, center_y = center_coords

        # Calculate the top-left corner of the laser image to be placed
        start_x = max(center_x - laser_w // 2, 0)
        start_y = max(center_y - laser_h // 2, 0)

        end_x = min(start_x + laser_w, face_w)
        end_y = min(start_y + laser_h, face_h)

        # Calculate the region of the face image to be merged with the laser image
        face_region_x = slice(start_x, end_x)
        face_region_y = slice(start_y, end_y)

        # Calculate the region of the laser image to be merged with the face image
        laser_region_x = slice(0, end_x - start_x)
        laser_region_y = slice(0, end_y - start_y)

        # Merge the cropped laser image with the face image using addWeighted
        for c in range(3):  # For each channel
            face_image[face_region_y, face_region_x, c] = \
                self.relu_like(
                    cv2.addWeighted(
                        face_image[face_region_y, face_region_x, c],
                        1,
                        laser_image[laser_region_y, laser_region_x, c],
                        self.laser_intensity,
                        0
                    ),
                    self.alpha
                )

        return face_image

    def adjust_brightness(self, image, face_image):
        """  Adjust the brightness of the image based on the power of the original face image"""
        P_o = np.mean(face_image)  # Power of the original face image
        P_a = self.laser_intensity  # Power of the laser intensity

        r_b = P_o / (P_o + P_a)

        # Adjust brightness
        adjusted_img = cv2.convertScaleAbs(image, alpha=r_b, beta=0)
        return adjusted_img

    def simulate_laser_attack(self, face_image_path, laser_images_path, center_coords, output_folder):
        """generate synthetic attackers by merging the laser images from laser_images_path
         and face images from face_image_path, the center coordinates need to be measured manually for each attacker."""

        # Load face image
        face_img = cv2.imread(face_image_path)

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over all laser images in the laser_images_path directory
        for laser_image_name in os.listdir(laser_images_path):
            laser_image_path = os.path.join(laser_images_path, laser_image_name)
            laser_img = cv2.imread(laser_image_path)

            if laser_img is None:
                continue  # Skip files that are not images

            # Resize laser image to match the width of the face image
            laser_img_resized = cv2.resize(laser_img, (
            face_img.shape[1], int(laser_img.shape[0] * face_img.shape[1] / laser_img.shape[1])))

            # Merge and adjust brightness
            merged_image = self.merge_laser_face(face_img.copy(), laser_img_resized, center_coords)
            final_image = self.adjust_brightness(merged_image, face_img)

            # Create output file name
            face_image_name = os.path.basename(face_image_path)
            output_image_name = f"{os.path.splitext(face_image_name)[0]}_{os.path.splitext(laser_image_name)[0]}.jpg"
            output_image_path = os.path.join(output_folder, output_image_name)

            # Save the result
            cv2.imwrite(output_image_path, final_image)

        print(f"Output images saved to: {output_folder}")


# Example usage
if __name__ == '__main__':

    face_image_path = '../data/attackers'
    laser_images_path = '../data/laser_images'
    center_coords = (545, 973)  # Center coordinates (x, y) of face image
    output_folder = '../data/synthetic_attackers'

    merger = LaserFaceMerger(alpha=255, laser_intensity=1.0)
    for face_name in os.listdir(face_image_path):
        face_folder_path = os.path.join(face_image_path, face_name)
        if os.path.isdir(face_folder_path):
            output_face_folder = os.path.join(output_folder, face_name)
            os.makedirs(output_face_folder, exist_ok=True)

            for face_image_name in os.listdir(face_folder_path):
                full_face_image_path = os.path.join(face_folder_path, face_image_name)
                merger.simulate_laser_attack(full_face_image_path, laser_images_path, center_coords, output_face_folder)
