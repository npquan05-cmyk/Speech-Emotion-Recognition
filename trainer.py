from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from .config import *
from .metrics import compute_metrics

def get_trainer(model, train_dataset, val_dataset, feature_extractor):

    data_collator = DataCollatorWithPadding(
        feature_extractor,
        padding=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,
        save_only_model=True,

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,

        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    return trainer
