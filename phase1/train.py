"""
Phase 1 - Step 4: Train the tiny CNN on the processed Speech Commands data.

Run:
    python train.py --data_dir ./data/processed --out_dir ./artifacts
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from features import featurize_file, get_feature_shape
from model import LABELS, NUM_CLASSES, build_tiny_ds_cnn

AUTOTUNE = tf.data.AUTOTUNE
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def list_files_and_labels(split_dir: Path):
    paths, labels = [], []
    for label in LABELS:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for wav in label_dir.glob("*.wav"):
            paths.append(str(wav))
            labels.append(LABEL_TO_IDX[label])
    return paths, labels


def make_dataset(paths, labels, batch_size=64, shuffle=False, augment=False):
    def _load(path, label):
        def _py_featurize(p):
            feat = featurize_file(p.numpy().decode("utf-8"))
            if augment:
                feat = feat + np.random.normal(0, 0.05, feat.shape).astype(np.float32)
            return feat

        feat = tf.py_function(_py_featurize, [path], tf.float32)
        feat.set_shape(get_feature_shape())
        feat = tf.expand_dims(feat, -1)  # add channel dim
        return feat, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels = list_files_and_labels(data_dir / "training")
    val_paths, val_labels = list_files_and_labels(data_dir / "validation")
    test_paths, test_labels = list_files_and_labels(data_dir / "testing")

    print(f"train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size)
    test_ds = make_dataset(test_paths, test_labels, args.batch_size)

    input_shape = get_feature_shape()
    model = build_tiny_ds_cnn(input_shape, NUM_CLASSES, width=args.width)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"), monitor="val_accuracy",
            save_best_only=True, mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Final held-out evaluation + FAR/FRR-relevant confusion matrix
    y_true, y_pred = [], []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    print("\nTest set report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))
    print("Confusion matrix (rows=true, cols=pred):")
    print(LABELS)
    print(confusion_matrix(y_true, y_pred))

    model.save(out_dir / "final_model.keras")
    print(f"\nSaved model to {out_dir / 'final_model.keras'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--width", type=float, default=1.0,
                         help="Channel width multiplier; try 0.5 if you need to shrink further")
    args = parser.parse_args()
    main(args)
