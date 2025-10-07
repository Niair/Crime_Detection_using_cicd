#### UCF Crime Dataset Download and Setup Script
==========================================

***This script automates the download and preparation of the UCF Crime dataset from Kaggle.
It handles API authentication, dataset download, extraction, and basic preprocessing.***

**Usage:**
    python scripts/download_data.py

**Requirements:**
    - kaggle package (pip install kaggle)
    - Your Kaggle API credentialspython scripts/download_data.py

**Dataset Overview**
- Source: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
- Download Date: Auto-generated
- Total Size: Check the raw/ directory
- Crime Categories: Multiple categories of criminal activities

**Directory Structure**
```
data/
|-- raw/                    # Original downloaded data
|   |-- videos/            # Video files
|   |-- annotations/       # Labels and annotations  
|   |-- metadata/          # Additional metadata
|-- processed/             # Processed data for training
    |-- train/             # Training data
    |-- val/               # Validation data  
    |-- test/              # Test data
```

**Usage Notes**
- Videos are in various formats (typically .mp4, .avi)
- Process videos into frames or features before training
- Split data appropriately for train/val/test

**Data Processing Steps**
1. Extract frames from videos
2. Resize frames to consistent dimensions  
3. Create train/validation/test splits
4. Generate feature vectors if needed

'''