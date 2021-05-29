# SIIM-Pneumothorax-Segmentation
Developed a model(Top 10% rank on Kaggle LB) that is able to segment/detect regions of Pneumothorax from chest X-rays it can help doctors with the diagnosis. This could aid in the early recognition of pneumothoraces and save lives.


## Dataset source:
Kaggle: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview

## Business Contrains
* The cost of misclassification can be high. We do not want a Pneumothorax person to be detected as normal.
* No strict latency constrain.

## Results
<img src="https://github.com/arunm8489/SIIM-Pneumothorax-Segmentation/blob/main/data/results.png" width=800 height=150>

## Final Model (ResUnet with additional DeepSupervision Block)
<img src="https://github.com/arunm8489/SIIM-Pneumothorax-Segmentation/blob/main/data/model.png" width=800 heigh=800>

Model is developed in Pytorch and is converted to ONNX runtime for faster inference. Inorder to conver trained model
```
python pytorch_to_onnx.py
```

### Usage

<img src="https://github.com/arunm8489/SIIM-Pneumothorax-Segmentation/blob/main/data/tree.png" height=300 width=400>

* models : Contains Pytorch as well as converted onnx model
* Notebooks: Contains Various training results using different models and parameters
* python_to_onnx.py: convert pytorchto onnx model
* utils.py: various helper functions
* test.py: code to test api



To create inference setup:

First install all required packages
```
pip3 install -r requirements.txt
```
Now start your server
```
python app.py
```

Prediction endpoint will be avalilable on http://localhost:4001/predict. 

**To test prediction yu can run test.py**
```
python test.py
```
