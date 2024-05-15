# LaserFR

## 1. Introduction

This is the official implementation code for the paper "The Invisible Polyjuice Potion: An Effective Physical Adversarial Attack Against Face Recognition." 
We aim to perform adversarial attacks on face recognition systems using an infrared diode embedded in the middle of glasses. 
First, we select the most likely pairs through filters and then use attack simulation to find the precise pairs and laser current. 
This guides the real-world attack to achieve a high success rate within a reasonable time.

## 2. Table of Contents

1. [Introduction](#1-introduction)
2. [Content Table](#2-content-table)
   - [Description](#21-description)
   - [Getting Started](#22-getting-started)
   - [Models](#23-models)
3. [Usage](#3-usage)
   - [Filters](#31-filters)
   - [Laser Signal Generation](#32-laser-signal-generation)
   - [Image Merge](#33-image-merge)
   - [Attack Simulation](#34-attack-simulation)
4. [Acknowledgement](#4-acknowledgment)

### 2.1 Project Structure

   #### Data
   - **data/**  
     Contains the original and simulated images of attackers and the targets.
   
   #### Models
   - **models/**  
     Stores the model parameters used in the project.
   
   #### Selected Data
   - **selected_data/**  
     Includes images that the filters have filtered out.
   
   #### Source Code
   - **src/**  
     Contains the source code for the project.
        The project includes four main parts corresponding to the paper:
      1. **Generating the Laser Model**: Implemented in `src/laser_generation.py`.
      2. **Merging Images**: Implemented in `src/image_merge.py`.
      3. **Filters Implementation**: Implemented in `src/filters.py`.
      4. **Attack Simulation**: Implemented in `src/face_recognition.py`.
   
   #### Tests
   - **tests/**  
     Includes tests that are used for the paper.


### 2.2 Getting Started

#### Prerequisites

- List any software or libraries that need to be installed before using the project.

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

   - For untargeted impersonation, place one attacker in the 'theOne' folder, and the targets in the 'theMany' folder. Run 'scr/filters.py'. The filtered targets will be copied to the 'selected_data' folder.
   
   - For targeted impersonation, place the targeted person in the 'theOne' folder. Run 'scr/filters.py'. The potential attackers will be copied to the 'selected_data' folder.
   
   - Multiple analyses at one time are supported. The corresponding results will be placed in subfolders named after the attackers/targets from the 'theOne' folder.

### 3.2 Laser signal generation

1. Input the parameters for the laser and camera in the 'laser_generation.py'.
   
2. Run the 'laser_generation.py' with the current value range and intervals.
   
3. The generated laser image named with current values will be saved in the 'data/laser_images' folder.
   

### 3.3 Image merge

1. After identifying the attacker, merge the attackers and the laser images to generate synthetic attack images.
   Currently, we can manually align the laser image with the attackers.
   We need to measure the coordinates of the center of the glasses and align the center of the laser with these coordinates.
   We hope to further develop an automatic merging process in the future.
   
3. The synthetic attackers named with the attackers' names and the current values will be saved in the 'data/synthetic_images' folder.


### 3.4 Attack simulation

1. Run 'src/face_recognition.py' to test the face recognition results with the synthetic attackers and the targets in the 'selected_data' folder from [3.1 Filters](#22-getting-started).
2. The results will be saved in the 'results.csv' file.
3. Get the successful attacker and target pair, and imformed laser's current range.

## 4. Acknowledgment

### 4.1 Attacker images on request

Due to IRB requirements, we cannot publicly share attacker images. If you need access, please contact us by email.

### 4.2 Citation





   


