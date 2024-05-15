# LaserFR

## 1. Introduction

This is the official implemental code for the paper 'The Invisible Polyjuice Potion: An Effective Physical Adversarial Attack Against Face Recognition'.

## 2. Table of Contents

1. [Introduction](#1-introduction)
2. [Content Table](#2-content-table)
   - [Description](#21-description)
   - [Getting Started](#22-getting-started)
   - [Models](#23-models)
3. [Usage](#3-usage)

### 2.1 Description

The project includes four main parts:
1. **Generating the Laser Model**: Implemented in `src/laser_generation.py`.
2. **Merging Images**: Implemented in `src/image_merge.py`.
3. **Filter Implementation**: Implemented in `src/filters.py`.
4. **Attack Simulation**: Implemented in `src/face_recognition.py`.

### 2.2 Getting Started

#### Prerequisites

- List any software or libraries that need to be installed before using the project.

#### Installation

Step-by-step instructions to install the necessary dependencies and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-title.git


## 3. Usage

### 3.1 Filters

1. Data Preparation
   
   - Put the attacker's original images in the data/attackers/ folder.
   
   - Put the targets's original images in the data/targets/ folder.

2. Run

   - For untargeted impersonation, place one attacker in the attackers folder, and the targets in the targets folder. Run 'scr/filters.py'. The filtered targets will be copied to the 'selected_data' folder.
   
   - For targeted impersonation, place the targeted person in the attacker folder. Run 'scr/filters.py'. The potential attackers will be copied to the 'selected_data' folder.
   
   - Multiple analyses at one time are supported. The corresponding results will be placed in subfolders named after the attackers/targets from the attacker folder.

### 3.2 Laser signal generation

1. Input the parameters for the camera.
   
2. Run the 'laser_generation.py'
   
3. The generated laser image under different current values will be saved in the 'data/laser_images' folder.
   

### 3.3 Image merge

1. After identifying the attacker, merge the attackers and the laser images to generate synthetic attack images.
   
2. The synthetic attackers will be saved in the 'data/synthetic_images' folder.


### 3.4 Attack simulation

1. Run 'src/face_recognition.py' to test the face recognition results with the synthetic attackers and the targets in the 'selected_data' folder from 3.1

2. Get the successful attacker and target pair and the laser current range.



   


