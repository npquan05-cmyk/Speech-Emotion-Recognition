from transformers import AutoModelForAudioClassification
from .config import *

id2label = {
    0:'neutral', 1:'calm', 2:'happy', 3:'sad',
    4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'
}

label2id = {v: k for k, v in id2label.items()}

def build_model():
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label
    )
    return model
