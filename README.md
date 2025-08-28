# Tracking objects with SAM2 (Segment Anything Model)

This repository uses demo code from [Facebook Research Team](https://github.com/facebookresearch/sam2) to track person in a given video.

## Hardware
CPU: intel Core i7 gen 10 \
GPU: NVIDIA RTX 3060 \ 
RAM: 64GB
Disk: 100GB SSD

## Software
OS: Windows 10 \
Python >= 3.8 \
CUDA >= 11.8 

## Main steps on this notebook
- Step 1: initialize environment and import needed libraries
- Step 2: Extract video into multiple frames
- Step 3: Mask the object in the first frame by bounding box around it and pass it to the model
- Step 4: The model will generate the additional mask for the next frame based on the current one, checking if mismatch masking
- Step 5: Merging frames into a video again  
