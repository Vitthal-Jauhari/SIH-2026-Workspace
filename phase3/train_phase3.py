"""
Phase 3 - Step 5: Training Tiny DS-CNN for Vaani Wake-Word Detection

Trains the 3-class tiny DS-CNN on:
- Augmented Vaani training clips (Ananya, Ark, Umang)
- Clean Speech Commands 'unknown' clips
- Clean ambient room 'silence' clips

Validation: Clean Ishita Vaani + validation negatives
Testing: Clean Unseen Vitthal Vaani + test negatives

Reuses Phase 1 model architecture and feature extraction.

Usage:
    # 2-epoch smoke test
    python train_phase3.py --data_dir ./data/combined --out_dir ./artifacts --epochs 2 --smoke_test

    # Full training
    python train_phase3.py --data_dir ./data/combined --out_dir ./artifacts --epochs 35
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Add phase1 to path to reuse model & features
PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import featurize_file, get_feature_shape  # noqa: E402
from model import LABELS, NUM_CLASSES, build_tiny_ds_cnn  # noqa: E402

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def load_dataset_features(split_dir: Path, cache_file: Path = None):
    """
    Extracts MFCC features for all files in a split directory.
    Caches features to disk as .npz for instant subsequent loads.
    """
    if cache_file and cache_file.exists():
        print(f"Loading cached features from {cache_file.name}...")
        data = np.load(cache_file)
        return data["x"], data["y"], list(data["paths"])

    paths, labels = [], []
    for label in LABELS:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            paths.append(str(wav))
            labels.append(LABEL_TO_IDX[label])

    print(f"Extracting features for {len(paths)} clips from {split_dir.name}...")
    features = []
    for p in paths:
        feat = featurize_file(p)
        features.append(feat)

    x = np.array(features, dtype=np.float32)  # (N, time, n_mfcc)
    x = np.expand_dims(x, -1)                 # (N, time, n_mfcc, 1)
    y = np.array(labels, dtype=np.int32)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y, paths=np.array(paths))

    return x, y, paths


def train_model(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 35,
    batch_size: int = 64,
    width: float = 1.0,
    smoke_test: bool = False,
    seed: int = 42,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = data_dir / ".feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = "smoke" if smoke_test else "full"
    print("=" * 65)
    print(f"PHASE 3 - STEP 5: {'SMOKE TEST' if smoke_test else 'FULL TRAINING'}")
    print("=" * 65)
    print(f"Data directory : {data_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Epochs         : {epochs}")
    print(f"Batch size     : {batch_size}")
    print(f"Channel width  : {width}\n")

    x_train, y_train, train_paths = load_dataset_features(
        data_dir / "training", cache_dir / f"train_feat_{width}.npz"
    )
    x_val, y_val, val_paths = load_dataset_features(
        data_dir / "validation", cache_dir / f"val_feat_{width}.npz"
    )
    x_test, y_test, test_paths = load_dataset_features(
        data_dir / "testing", cache_dir / f"test_feat_{width}.npz"
    )

    print(f"Training set   : {x_train.shape[0]} samples (shape: {x_train.shape[1:]})")
    print(f"Validation set : {x_val.shape[0]} samples")
    print(f"Testing set    : {x_test.shape[0]} samples\n")

    input_shape = get_feature_shape()
    model = build_tiny_ds_cnn(input_shape, NUM_CLASSES, width=width)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks if not smoke_test else [],
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    preds = model.predict(x_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    print("\n" + "=" * 65)
    print("TEST EVALUATION REPORT:")
    print("=" * 65)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(LABELS)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save final model
    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)
    print(f"\nSaved model checkpoint to {final_model_path}")

    # Save training metadata
    meta = {
        "smoke_test": smoke_test,
        "epochs_trained": len(history.history["loss"]),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / f"training_metadata_{tag}.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"[SUCCESS] Training complete. Metadata saved to {out_dir / f'training_metadata_{tag}.json'}\n")
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/combined")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--smoke_test", action="store_true", help="Run 2-epoch smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        Path(args.data_dir),
        Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        width=args.width,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
