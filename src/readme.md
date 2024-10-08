## 1. Laser Generator

This script simulates a laser attack on a CMOS sensor by generating an interference pattern and converting the resulting power distribution into RGB values based on the sensor's quantum efficiency.
Input the parameters for the laser and camera in the `laser_generation.py`. The parameters used in the paper are listed in the following table.
   
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

After running the script, the generated laser images named with current values will be saved in the `data/laser_images` folder. Manually verify the output images to ensure they meet the expected simulation results.

## 2. Image Merge

This script simulates a laser attack on face images by merging laser images onto the face images and adjusting the brightness accordingly.

### Example Usage

- **`if __name__ == '__main__':`**:
  - `face_image_path`: Directory containing face images. `data/attackers_original` by default.
  - `laser_images_path`: Directory containing laser images. `data/laser_images` by default.
  - `center_coords`: Center coordinates (x, y) of the face image means the turned-off laser's coordinates of the attacker.
    
     Currently, we need to align the laser image with the attackers manually.
     We need to measure the coordinates of the glasses' center and align the laser's center with these coordinates.
     We hope to further develop an automatic merging process in the future.
  - `output_folder`: Output directory for synthetic images. `data/synthetic_attackers` by default.
  
  After running the script, the synthetic attackers named with the attackers' names and the current values will be saved in the `data/synthetic_attackers` folder.

## 3. Filters

This script conducts the ES filter and PSAS filter. So that we do not need to run the simulation attack with all attackers and targets.

   - For untargeted impersonation, the `theOne` path points to the attacker, and the `theMany` points to targets. Run `scr/filters.py`. If the output is not None, the untargeted attack will succeed, and the results will be saved for the predictable untargeted attack.
   
   - For targeted impersonation, the `theOne` path points to the target, and the `theMany` points to the attackers. Run `scr/filters.py`. And the results will be saved for the following simulation attack.
   
   - Multiple analyses at one time are supported. The corresponding results will be placed in subfolders named after the attackers/targets from the `theOne` folder.

### Example Usage

- **`if __name__ == '__main__':`**:
  - Creates an instance of the `ExplanationGenerator` class.
  - For the targeted attack, set the `theOne' path pointing to `data/theOne' where placing the selected target. If want to test with multiple targets, point the path to the targets (e.g., data/I-50). The `theMany' path points to the attackers available.
  - For the untargeted attack, set the `theOne1' path pointing to `data/theOne' where placing the selected attacker. If want to test with multiple attackers, point the path to the attacker's set (e.g., data/attackers). The `theMany1' path points to the available attackers.
  - Results in CSV files are saved for further analysis, `/results/targeted_pairs.csv` and `/results/untargeted_pairs.csv`, respectively with the model name as a prefix.

   If the `move` option of the ES filter is True, a folder named `es_selected_images` will be created to save the filtered images. 
   Similarly, if the `move` option of the PSAS filter is True, a folder named `psas_selected_images` will be created to save the filtered images. 

   By default, the option for the ES filter is set to False, while the PSAS filter is set to True. 

### Model and Weights

The ExplanationGenerator class relies on pre-trained models (e.g., ResNet152 or ResNet34). The customized models and parameters are saved in the `Models' folder. We just adapted the models used in our paper. If you want to test more models, please refer to the [original paper](https://arxiv.org/pdf/1909.12977) to adapt the models and parameters.

## 4. Face Recognition Attack Analysis

- This project involves analyzing face recognition results to inform impersonation attacks using the DeepFace library. The script iterates over images of synthetic attackers and compares them with images of targets to find matches.
When running the Python script for the first time, it will cost plenty of time. As we claimed, it time-cost task to do the embedding calculation. So the deepface platform will save the embeddings for each iamge for quick use in the following test. If the dataset is changed, it will need to recalculate and save the embeddings.

- The untargeted attacks against 4 models with 2 preprocessing methods and 3 distance metrics will be tested, and the results will be saved in the `results/` folder tagged with the combinations. RetinaFace will raise unpredictable errors when tested recently, we removed it temporarily and will add it back in the later updates.

- The targeted attack against ArcFace is conducted. The results will be saved in the `targeted_impersonation_all_results.csv`. Meanwhile, the matching results compared with the filters are also conducted and the results are saved in the `targeted_impersonation_matching_results.csv`. 




