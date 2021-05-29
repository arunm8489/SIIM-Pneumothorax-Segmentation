from flask import Flask, request, jsonify
from utils import *
import yaml, json
import cv2,os
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

with open('config.yaml') as yaml_file:
    config = yaml.safe_load(yaml_file)

config = config['CONFIGS']
model_path = config['MODEL_PATH']
img_size = config['IMG_SIZE']
device = config['DEVICE']
mask_thresh = config['MASK_THRESHOLD']
pred_thresh = config['PRED_THRESHOLD']
min_area = config['MIN_AREA']

MODEL = ort.InferenceSession(model_path)
transform = get_transforms()



def get_prediction(image):
    """
    model prediction
    returns mask,probability
	
    """
    image = image.transpose(2,0,1)
    batch = np.expand_dims(image,axis=0)
    onix_input = {"in_image": batch}
    mask,prob = MODEL.run(None, onix_input)
    return mask,prob

@app.route('/predict',methods=['POST'])
def predict_img():
    data = json.loads(request.data)
    bs64_image = data['image']
    try:
        image = decode_img(bs64_image)
        # preprocessing
        print(image.shape) 
        out = transform(image=image)
        image = out["image"]
        mask,prob = get_prediction(image)
        mask,prob = sigmoid(mask[0][0]),sigmoid(prob[0][0])
        mask = post_process_mask(mask,prob,mask_thresh=mask_thresh,min_area=min_area,pred_thres=pred_thresh)
        mask = encode_img(mask)
        data = {
            "image": mask,
            "probability": str(prob)
        }
    except Exception as e:
        error = {"error": str(e)}
        return jsonify(success="false",data=error)
    else:
        print(data)
        return jsonify(success="true", data=data)
    


if __name__== "__main__":
   app.run(port=4001)

    