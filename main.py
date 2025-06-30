# main.py

from scripts.preprocess import load_and_prepare_data
from scripts.model import train_and_evaluate_model

def main():
    print("📥 Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    print("🧠 Training and evaluating model...")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)

    print("✅ Done! Results saved in 'outputs/' folder.")

if __name__ == "__main__":
    main()
