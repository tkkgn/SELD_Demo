# User guide for demo
3 new files was created to do animation for this model:
- web_demo.py: main source for web app
- Scatter_Animation.py: create animation for dataframe
- storage_params.py: store the parameter for dumping result and animation 

## Installation
In this demo we use Python 3.10 instead of the default Python_Version of model 3.7.

Install all required dependencies run:
```bash
pip install -r requirements.txt
```
## Setup storage 
This is optional, you can set up your own way. However, you must remember to update it in storage_params.py

**NOTE**: The following commands are using with Ubuntu OS(linux).
**The "SELD_Demo/" is current folder**
Make main directory path:
```bash
mkdir demo_With_Streamlit
mkdir demo_With_Streamlit/storage
mkdir demo_With_Streamlit/backround
```
## Download background:
**The "SELD_Demo/" is current folder**

```bash
cd demo_With_Streamlit/backround/
```
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dt0cFZXV5td9090lRjuNSHeJNI8x5AyQ' -O sample_room.png
```

* [Click here](https://drive.google.com/file/d/1dt0cFZXV5td9090lRjuNSHeJNI8x5AyQ/view?usp=sharing) for manual download backround

## Download model:
**The "SELD_Demo/" is current folder**

Create directory to save the model weight 
```bash
mkdir RESULTS/Task2
cd RESULTS/Task2
```
Download model weight
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gno2EONz2q9aPENkztIxcTkRdrXw3CkJ' -O baseline_task2_checkpoint
```
* [Click here](https://drive.google.com/file/d/1gno2EONz2q9aPENkztIxcTkRdrXw3CkJ/view) for manual download model weight


## Running program

To run whole program using:
```bash
streamlit run web_demo.py
```
**NOTE**: You should use "L3DAS22 Dataset(default)" for better results. Please assure browsing a correct input with your dataset option if the errors pop up.


