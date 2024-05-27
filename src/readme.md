# Laser Generator

This script simulates a laser attack on a CMOS sensor by generating an interference pattern and converting the resulting power distribution into RGB values based on the quantum efficiency of the sensor.

## Important Notes

### Performance

Depending on the resolution and number of calculations, the script may take some time to complete. Ensure you have sufficient computational resources.

### Verification

After running the script, manually verify the output images to ensure they meet the expected simulation results.

# Image Merge

This script simulates a laser attack on face images by merging laser images onto the face images and adjusting the brightness accordingly.

## Example Usage

#### Paths:

- `face_image_path`: Directory containing face images.
- `laser_images_path`: Directory containing laser images.
- `center_coords`: Center coordinates (x, y) of the face image means the turned-off laser's coordinates of the attacker.
- `output_folder`: Output directory for synthetic images.

#### Simulation:

1. An instance of `LaserFaceMerger` is created with specified alpha and laser intensity values.
2. The `simulate_laser_attack` method is called for each face image in the specified directory to simulate the laser attack and save the results in the output directory.

# Filters

This script conducts the ES filter and PSAS filter.

## Example Usage



## Important Notes

### Directory Structure

Ensure that the directories for target images (theOne) and attacker images (theMany) are correctly specified and contain images in supported formats (JPEG, PNG).

### Model and Weights

The ExplanationGenerator class relies on pre-trained models (e.g., ResNet152 or ResNet34). Ensure that the model weights are available at the specified path (Models/parameters.pth).
