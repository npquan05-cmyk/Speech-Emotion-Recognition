from src.config import *
from src.data_prepare import prepare_ravdess_df
from src.preprocess import load_and_preprocess
from src.model import build_model
from src.trainer import get_trainer
from src.visualize import plot_loss

def main():

    # 1. Load data
    df = prepare_ravdess_df(ROOT_PATH)
    print("Số lượng sample:", len(df))

    # 2. Preprocess
    dataset, feature_extractor = load_and_preprocess(df)

    split_data = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_dataset = split_data["train"].shuffle(seed=SEED)
    val_dataset = split_data["test"]

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))

    # 3. Model
    model = build_model()

    # 4. Trainer
    trainer = get_trainer(model, train_dataset, val_dataset, feature_extractor)

    # 5. Train
    print("🚀 Bắt đầu training...")
    trainer.train()

    # 6. Save
    trainer.save_model(BEST_MODEL_DIR)
    feature_extractor.save_pretrained(BEST_MODEL_DIR)

    # 7. Plot
    plot_loss(trainer.state.log_history)


if __name__ == "__main__":
    main()
