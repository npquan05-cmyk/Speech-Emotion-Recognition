import os
import pandas as pd

def prepare_ravdess_df(root_path):
    data = []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".wav"):
                part = file.split('.')[0].split('-')
                if len(part) >= 3:
                    label = int(part[2]) - 1
                    data.append({
                        "path": os.path.join(root, file),
                        "label": label
                    })

    return pd.DataFrame(data)
