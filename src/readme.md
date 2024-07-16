### 3.1 Filters

1. Data Preparation
   
   - Prepare both the attackers' and targets' original image datasets.
   
   - There are two subfolders within the data folder:
      - **theOne/**: Contains images of a specific attacker or target.
      - **theMany/**: Contains images that need to be filtered.

2. Run

   - For untargeted impersonation, place one attacker in the `theOne` folder, and the targets in the `theMany` folder. Run `scr/filters.py`. The filtered targets will be copied to the `selected_data` folder.
   
   - For targeted impersonation, place the targeted person in the `theOne` folder, and the attackers in the `theMany` folder. Run `scr/filters.py`. The potential attackers will be copied to the `selected_data` folder.
   
   - Multiple analyses at one time are supported. The corresponding results will be placed in subfolders named after the attackers/targets from the `theOne` folder.

### 3.2 Laser signal generation

1. Input the parameters for the laser and camera in the `laser_generation.py`.

   
      | Parameter        | Description                                   | Example Value         |
      |------------------|-----------------------------------------------|-----------------------|
      | **LaserModel**   |                                               |                       |
      | `P`              | Power of the laser in mW                      | 100.0                 |
      | `wavelength`     | Wavelength of the laser light in meters       | 0.000000785 (785 nm)  |
      | `theta`          | Divergence angle in radians                   | 0.028                 |
      | `n`              | Refractive index                              | 1.5                   |
      | `d`              | Distance from aperture to the lens in meters  | 0.004                 |
      | `f`              | Focal length in meters                        | 0.013                 |
      | `z`              | Distance from the beam waist in meters        | 0.35                  |
      | `t`              | Lens thickness in meters                      | 0.003                 |
      | `r_a`            | Aperture radius in meters                     | 0.0072                |
      | **CMOSSensor**   |                                               |                       |
      | `width`          | Width of the CMOS sensor in meters            | 0.004                 |
      | `height`         | Height of the CMOS sensor in meters           | 0.003                 |
      | `pixel_size`     | Size of a single pixel in meters              | 1e-6 (1.0 µm)         |
      | `QE_r`           | Quantum Efficiency for the red channel        | 0.33                  |
      | `QE_g`           | Quantum Efficiency for the green channel      | 0.2                   |
      | `QE_b`           | Quantum Efficiency for the blue channel       | 0.08                  |
      | `exposure_time`  | Exposure time in seconds                      | 1.0/30                |

   
2. Run the `laser_generation.py` with the current value range and intervals.
   
3. The generated laser image named with current values will be saved in the `data/laser_images` folder.
   

### 3.3 Image merge

1. After identifying the attacker, merge the attackers and the laser images to generate synthetic attack images.
   
   Currently, we need to align the laser image with the attackers manually.
   We need to measure the coordinates of the glasses' center and align the laser's center with these coordinates.
   We hope to further develop an automatic merging process in the future.
   
3. The synthetic attackers named with the attackers' names and the current values will be saved in the `data/synthetic_attackers` folder.


### 3.4 Attack simulation

1. Run `src/face_recognition.py` to test the face recognition results with the synthetic attackers and the targets in the `selected_data` folder from [3.1 Filters](#22-getting-started). (Sections 6.3 and 6.4)
   
2. The results will be saved in the `results/impersonation_results.csv` file.

   The results contain 4 columns, 'identity' is the targets identified, 'attacker' is the attacker's name, and 'laser setting' is the informed laser current range.
   
### 3.5 Other Tests

1. Run `tests/celebritie_recognition.py` to test the black-box attack against Amazon Recognition. (Section 6.6)
   
   To run this test, you need to configure your Boto3 credentials on AWS. You can refer to the [official document](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html). The results will be saved in `results/celeb_results.csv`.

2. Run `tests/continual_attack.py` to test the continual attack. (Section 7.1)

3. Run `tests/dodging.py` to test the dodging/DoS attack. (Section 6.2)
   
4. Run `tests/real_timeFR.py` to test the attack in the real world.

# 1. Laser Generator

This script simulates a laser attack on a CMOS sensor by generating an interference pattern and converting the resulting power distribution into RGB values based on the sensor's quantum efficiency.

## Important Notes

### Performance

Depending on the resolution and number of calculations, the script may take some time to complete. Ensure you have sufficient computational resources.

### Verification

After running the script, manually verify the output images to ensure they meet the expected simulation results.

# 2. Image Merge

This script simulates a laser attack on face images by merging laser images onto the face images and adjusting the brightness accordingly.

## Example Usage

- **`if __name__ == '__main__':`**:
  - `face_image_path`: Directory containing face images.
  - `laser_images_path`: Directory containing laser images.
  - `center_coords`: Center coordinates (x, y) of the face image means the turned-off laser's coordinates of the attacker.
  - `output_folder`: Output directory for synthetic images.

  #### Simulation:
  
  1. An instance of `LaserFaceMerger` is created with specified alpha and laser intensity values.
  2. The `simulate_laser_attack` method is called for each face image in the specified directory to simulate the laser attack and save the results in the output directory.

# 3. Filters

This script conducts the ES filter and PSAS filter.

## Example Usage

- **`if __name__ == '__main__':`**:
  - Creates an instance of the `ExplanationGenerator` class.
  - Calls the `es_filter` method to filter theMany images.
  - Calls the `psas_filter` method to filter the images got by es_filter.
  - Save the results in a CSV file for further analysis.

If the `move` option of ES filter is True, a folder named `es_selected_images` will be created to save the filtered images. 
Similarly, if the `move` option of PSAS filter is True, a folder named `psas_selected_images` will be created to save the filtered images. 

By default, the option for the ES filter is set to False, while the PSAS filter is set to True. This is because we will use the images in the `psas_selected_images` folder to run simulated attacks.

## Important Notes

### Directory Structure

Ensure that the directories for target images (theOne) and attacker images (theMany) are correctly specified and contain images in supported formats (JPEG, PNG).

### Model and Weights

The ExplanationGenerator class relies on pre-trained models (e.g., ResNet152 or ResNet34). Ensure that the model weights are available at the specified path (Models/parameters.pth).

# 4. Face Recognition Attack Analysis

This project involves analyzing face recognition results to inform impersonation attacks using the DeepFace library. The script iterates over images of synthetic attackers and compares them with images of targets to find matches.



