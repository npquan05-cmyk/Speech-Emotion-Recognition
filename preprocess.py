from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor
from .config import *

def load_and_preprocess(df):

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["path"]]

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            truncation=True,
            max_length=SAMPLING_RATE * MAX_DURATION
        )

        inputs["labels"] = list(examples["label"])
        return inputs

    encoded_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=8
    )

    return encoded_dataset, feature_extractor
