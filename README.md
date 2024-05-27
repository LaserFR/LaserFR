# LaserFR

## 1. Introduction

This is the official implementation code for the paper "The Invisible Polyjuice Potion: An Effective Physical Adversarial Attack Against Face Recognition." 
We aim to perform adversarial attacks on face recognition systems using an infrared diode embedded in the middle of glasses. 
First, we select the most likely pairs through filters and then use attack simulation to find the precise pairs and laser current. 
This guides the real-world attack to achieve a high success rate within a reasonable time.

## 2. Table of Contents

1. [Introduction](#1-introduction)
2. [Content Table](#2-content-table)
   - [Description](#21-project-structure)
   - [Getting Started](#22-getting-started)
3. [Usage](#3-usage)
   - [Filters](#31-filters)
   - [Laser Signal Generation](#32-laser-signal-generation)
   - [Image Merge](#33-image-merge)
   - [Attack Simulation](#34-attack-simulation)
   - [Other attacks](#35-other-tests)
4. [Acknowledgement](#4-acknowledgment)

### 2.1 Project Structure

   - **data/**  
     Contains the original and simulated images of attackers and the the original targets.
     And the laser images and the real-world captured images.
   
   - **models/**  
     Stores the model parameters used in the project.
   
   - **selected_data/**  
     Includes images that the filters have filtered out.

   - **results/**  
     Save the results for different attacks.
   
   - **src/**  
     Contains the source code for the project.
        The project includes four main parts corresponding to the paper:
      1. **Generating the Laser Model**: Implemented in `laser_generation.py`.
      2. **Merging Images**: Implemented in `image_merge.py`.
      3. **Filters Implementation**: Implemented in `filters.py`.
      4. **Attack Simulation**: Implemented in `face_recognition.py`.
   
   - **tests/**  
     Includes tests that are used for the paper.
      1. **Black-box Test**: Implemented in `celebritie_recognition.py`.
      2. **Continual Attack**: Implemented in `continual_attack.py`.
      3. **DoS and Dodging**: Implemented in `dodgings.py`.
      4. **Physically Real-time Test**: Implemented in `real_timeFR.py`.


### 2.2 Getting Started

#### Prerequisites

- We tested our attack using [FaceNet](https://github.com/davidsandberg/facenet), [facenet-pytorch](https://github.com/timesler/facenet-pytorch), and [deepface](https://github.com/serengil/deepface).
  Please install these libraries first by running the following command:
  ```bash
   pip install facenet facenet-pytorch deepface

#### Installation

Step-by-step instructions to install the necessary dependencies and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-title.git
2. Navigate to the project directory:
   ```bash
   cd project-title
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## 3. Usage

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

1. Run `src/face_recognition.py` to test the face recognition results with the synthetic attackers and the targets in the `selected_data` folder from [3.1 Filters](#22-getting-started).
   
2. The results will be saved in the `results/impersonation_results.csv` file.

   The results contain 4 columns, 'identity' is the targets identified, 'attacker' is the attacker's name, and 'laser setting' is the informed laser current range.
   
### 3.5 Other Tests

1. Run `tests/celebritie_recognition.py` to test the black-box attack against Amazon Recognition.
   
   To run this test, you need to configure your Boto3 credentials on AWS. You can refer to the [official document](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html). The results will be saved in `results/celeb_results.csv`.

2. Run `tests/continual_attack.py` to test the continual attack.

3. Run `tests/dodging.py` to test the dodging/DoS attack.
   
4. Run `tests/real_timeFR.py` to test the attack in the real world.
  

## 4. Acknowledgment

### 4.1 Attacker images on request

Due to IRB requirements, we cannot publicly share attacker images. If you need access, please contact us by email.

### 4.2 Citation







   


