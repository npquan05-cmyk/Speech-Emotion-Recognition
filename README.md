# 🎧 Speech Emotion Recognition using Wav2Vec2

## 🚀 Overview

Dự án này xây dựng hệ thống **Speech Emotion Recognition (SER)** sử dụng mô hình **Wav2Vec2** từ HuggingFace để phân loại cảm xúc trong giọng nói.

Pipeline bao gồm:

* Chuẩn bị dữ liệu từ file `.wav`
* Tiền xử lý audio (feature extraction)
* Fine-tune mô hình Wav2Vec2
* Huấn luyện với HuggingFace Trainer
* Đánh giá bằng Accuracy và F1-score
* Lưu model tốt nhất
* Vẽ biểu đồ loss (train vs validation)

---

## 📂 Dataset

Dataset sử dụng: **RAVDESS Emotional Speech Audio**

Cấu trúc dữ liệu:

* Các file `.wav`
* Label được trích xuất từ tên file

### Cách extract label:

```python
label = int(part[2]) - 1
```

### Mapping label:

| ID | Emotion   |
| -- | --------- |
| 0  | neutral   |
| 1  | calm      |
| 2  | happy     |
| 3  | sad       |
| 4  | angry     |
| 5  | fearful   |
| 6  | disgust   |
| 7  | surprised |

---

## ⚙️ Data Preparation

* Duyệt toàn bộ thư mục bằng `os.walk`
* Tạo DataFrame gồm:

  * `path`: đường dẫn file audio
  * `label`: nhãn cảm xúc

Chuyển sang HuggingFace Dataset:

```python
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
```

---

## 🎧 Preprocessing

Sử dụng:

```python
AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
```

### Xử lý:

* Resample về **16kHz**
* Truncate tối đa **5 giây**
* Không padding tại bước preprocess

```python
max_length = 16000 * 5
```

---

## ✂️ Train / Validation Split

```python
test_size = 0.2
seed = 42
```

* Train: 80%
* Validation: 20%

---

## 🧠 Model

Sử dụng:

```python
AutoModelForAudioClassification
```

### Cấu hình:

* Pretrained: `facebook/wav2vec2-base-960h`
* Num labels: 8
* Mapping:

  * `id2label`
  * `label2id`

---

## 📊 Metrics

Sử dụng:

* Accuracy (HuggingFace evaluate)
* F1-score (weighted)

```python
f1_score(labels, preds, average="weighted")
```

---

## 🧩 Data Collator

```python
DataCollatorWithPadding(feature_extractor, padding=True)
```

👉 Đây là phần **quan trọng để fix lỗi padding** khi training.

---

## ⚙️ Training Configuration

### TrainingArguments:

* Epochs: **20**
* Batch size: **8**
* Gradient Accumulation: **2**
* Learning rate: **2e-5**
* Scheduler: **Cosine**
* Warmup: **0.1**

### Optimization:

* FP16 (giảm memory)
* EarlyStopping (patience = 3)

### Checkpoint:

* Lưu theo epoch
* Chỉ giữ **2 checkpoint**
* Không lưu optimizer (giảm dung lượng)

```python
save_total_limit=2
save_only_model=True
```

---

## 🏋️ Training

```bash
python main.py
```

Trong quá trình train:

* Log mỗi 10 steps
* Evaluate mỗi epoch
* Load best model theo F1-score

---

## 💾 Save Model

Sau khi train:

```python
trainer.save_model("./best_model")
feature_extractor.save_pretrained("./best_model")
```

---

## 📉 Visualization

Vẽ biểu đồ:

* Train Loss (average theo epoch)
* Validation Loss

```python
plot_loss(trainer.state.log_history)
```

---

## 📦 Requirements

```txt
transformers
datasets
evaluate
numpy
pandas
matplotlib
scikit-learn
torch
```

---

## ⚠️ Lưu ý quan trọng

* Dataset lớn → không push lên GitHub
* Nên sử dụng GPU để train
* Không padding trong preprocess (đã xử lý bằng DataCollator)
* Giới hạn độ dài audio (5 giây) để tránh OOM

---

## 🔍 Key Features

* ✅ Fine-tune Wav2Vec2 cho bài toán SER
* ✅ Pipeline chuẩn HuggingFace Trainer
* ✅ Dynamic padding (fix lỗi phổ biến)
* ✅ Early stopping + checkpoint tối ưu
* ✅ Visualization training process

---

## 👨‍💻 Author

Nguyễn Phú Quân

---

## 📜 License

Dự án phục vụ mục đích học tập và nghiên cứu.
