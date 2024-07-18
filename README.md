# LaserFR

## 1. Introduction

This is the official implementation code for the paper "The Invisible Polyjuice Potion: An Effective Physical Adversarial Attack Against Face Recognition."

Our work involves modeling physical infrared lasers and using simulation results to guide physical attacks. The structure of our project mirrors that of the paper, where a Python script corresponds to a step. This approach facilitates step-by-step verification of our methodology and results, and reuse and repurposing by others.
As shown in the figure:
![workflow](https://github.com/LaserFR/LaserFR/blob/main/images/github%20overflow.png)

Additionally, we provide a main Python file (main.py) that integrates all approaches into an integrated execution. This file allows the complete execution of a desired attack scenario against the FR models under study.

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
     Stores the FR models and corresponding parameters used in the project.
   
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

   - **images/**
     
     It includes images and videos that show real-world approaches and results.
         
   Each directory contains a detailed README file that explains its usage. Additionally, each code file includes comments describing the code's functions and operations.

### 2.2 Getting Started

#### Prerequisites

We run our test under Windows 10. The code can run with just the CPU. If you have a GPU, the code may be accelerated.

#### Installation

Step-by-step instructions to install the necessary dependencies and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/LaserFR/LaserFR.git
2. Navigate to the project directory:
   ```bash
   cd ./LaserFR
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

 As we tested our attack using [FaceNet](https://github.com/davidsandberg/facenet), [facenet-pytorch](https://github.com/timesler/facenet-pytorch), and [deepface](https://github.com/serengil/deepface). Please ensure these libraries have the same version listed in requirements.txt, as we noticed that the deepface just released a significant version update.


## 3. Usage

### 3.1 Prepare for the face dataset

1. prepare the targeted face dataset
   
    Run `LFWPicker.py`, it will download the [lwf_funneled dataset](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz), and generate the I-K dataset, which randomly picks K 
    identities from the LFW dataset. the original LFW dataset and the I-K dataset will be saved in the `data/` folder. If the download link does not work, you can manually 
    download the dataset and set the parameter of LFWPicker `dataset_path='path to existing lfw'`.

2. prepare the attacker dataset

   We need to take the attackers' original photo wearing the customized glasses, the laser is turned off. And then put the images in the `data/attackers`. The attackers' images should be placed under the folder named after the attacker.

   
### 3.2 Genenerate laser images

Based on the parameters of the targeted camera and the infrared laser used, laser images can be generated under different laser powers.

   - Implementation of Section 5.1 and achieved in `laser_generation.py`.
     
   - Run `laser_generation.py`, the generated laser images will be saved in the `data/laser_images/`. You can check them here and the images are named ending with power value。 

### 3.2 Image merge

Merge the attackers and the laser images to generate synthetic attack images.

   - Implementation of Section 5.2 and achieved in `image_merge.py`.

   - Run `image_merge.py`, the attackers in the `data/attackers` will be merged with the laser images from the previous approach. The synthetic attack images will be saved in `data/synthetic_attackers/`.

### 3.3 Filters

Verify if an untargeted attack can succeed without running a synthetic attack, and select the predictable target for predictable untargeted impersonation and the optimal attacker for a targeted impersonation attack with reduced computation workload.

   - Implementation of enhanced research of Section 5.3.4 and achieved in `filters.py`.

   - Run `filters.py`. For targeted impersonation, if the target is only one, the selected attackers for the target will be printed in the terminal. The results will be saved as `targeted_pairs.csv` in the `results` folder. If choose to move the images, the filtered-out images will be copied to `selected_data`.
   - For untargeted impersonation, attackers who can achieve the untargeted attack will be printed in the terminal, and The results will be saved as `untargeted_pairs.csv` in the `results` folder. If choose to move the images, the filtered-out images will be copied to `selected_data`.


### 3.4 Attack simulation

Verify if an untargeted attack can succeed without running a synthetic attack, and select the predictable target for predictable untargeted impersonation and the optimal attacker for a targeted impersonation attack with reduced computation workload.

   - Implementation of attack simulation of Section 5.3, and test for Section 6.3, Section 6.4, and Section 7.2. And achieved in `face_recognition.py`.
     
   - The results will be saved in the `results/***_impersonation_results.csv` file.

   
### 3.5 Other Tests

1. Run `tests/celebritie_recognition.py` to test the black-box attack against Amazon Recognition. (Section 6.6)
   
   To run this test, you need to configure your Boto3 credentials on AWS. You can refer to the [official document](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html). 

2. Run `tests/continual_attack.py` to test the continual attack. (Section 7.1)

3. Run `tests/dodging.py` to test the dodging/DoS attack. (Section 6.2)
   
4. Run `tests/real_timeFR.py` to test the attack in the real world.
  

## 4. Acknowledgment

### 4.1 Attacker images on request

Due to IRB requirements, we cannot publicly share attacker images. If you need access, please feel free to contact us by email.

### 4.2 References

https://github.com/Jeff-Zilence/Explain_Metric_Learning

https://github.com/davidsandberg/facenet

https://github.com/timesler/facenet-pytorch

https://github.com/serengil/deepface








   


