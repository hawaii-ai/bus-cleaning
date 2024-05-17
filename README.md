# BUS Cleaning
---
## Paper Title
###### Github repository containing all relevant code for journal submission
This repository is designed to provide an open-source solution for cropping BUS images for injestion into deep learning pipelines, flagging scans with abnormalities not consistently indicated in BUS DICOM metadata, and extracting knowledge about scan position and purpose from burnt-in scan annotations. 

### Pipeline Overview:
Insert diagram showing full pipeline here

### Results
Insert table of performance results here 

## Installation and system requirements
- Tested on Ubuntu 20.04.6 LTS
- Python version: 3.9.16
- To install dependencies, run:
```python3
python setup.py install
```
## Demo
- Demo scripts are provided in the outermost folder.
- A demo dataset is provided purely to validate code functionality, the dataset is not representative of the complete dataset used to develop or evaluate pipeline performance in the manuscript. 
- To validate code functionality, run sample code in notebook corresponding to desired functionality (e.g. for illustrative examples of flagging and cropping procedures)
	- Kailee todo: fill in 
    - `.ipynb` for sample script
    - `.py` for sample script