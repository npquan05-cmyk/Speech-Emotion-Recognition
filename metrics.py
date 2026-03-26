import numpy as np
import evaluate
from sklearn.metrics import f1_score

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1
    }
