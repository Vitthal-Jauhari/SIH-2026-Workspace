"""
Phase 3 - Step 5: Evaluate the retrained model broken down BY SPEAKER.

The headline number this script exists to produce is the gap between
accuracy on speakers the model trained on vs. the speakers it never heard --
that gap is the real measure of whether Phase 3 generalizes, not the
aggregate accuracy (which Phase 2's eval_wav.py already gives you).

Run:
    python eval_by_speaker.py \
        --tflite_path ./artifacts/vikramedge_phase3_int8.tflite \
        --speakers_dir ./data/speakers \
        --held_out_speakers charlie dana
"""

import argparse
import sys
from pathlib import Path

import numpy as np

for _cand in [Path(__file__).resolve().parent.parent / "phase2", Path(__file__).resolve().parent.parent / "vikramedge_phase2", Path(__file__).resolve().parent]:
    if (_cand / "inference.py").exists():
        sys.path.insert(0, str(_cand))
        break
from inference import WakeWordModel  # noqa: E402


def main(args):
    model = WakeWordModel(args.tflite_path)
    speakers_dir = Path(args.speakers_dir)
    held_out = set(args.held_out_speakers)

    per_speaker = {}
    for speaker_dir in sorted(speakers_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        correct, total = 0, 0
        for word_dir in speaker_dir.iterdir():
            true_label = word_dir.name
            for wav in word_dir.glob("*.wav"):
                pred_label, probs, _ = model.predict_from_file(str(wav))
                total += 1
                correct += int(pred_label == true_label)
        if total:
            per_speaker[speaker_dir.name] = (correct, total)

    print(f"{'Speaker':<15}{'Accuracy':<12}{'N':<6}{'Seen/Unseen'}")
    print("-" * 45)
    seen_accs, unseen_accs = [], []
    for speaker, (correct, total) in per_speaker.items():
        acc = correct / total
        status = "UNSEEN" if speaker in held_out else "seen"
        (unseen_accs if speaker in held_out else seen_accs).append(acc)
        print(f"{speaker:<15}{acc*100:>6.1f}%     {total:<6}{status}")

    print("-" * 45)
    if seen_accs:
        print(f"Mean accuracy, seen speakers:   {np.mean(seen_accs)*100:.1f}%")
    if unseen_accs:
        print(f"Mean accuracy, UNSEEN speakers:  {np.mean(unseen_accs)*100:.1f}%")
    if seen_accs and unseen_accs:
        gap = (np.mean(seen_accs) - np.mean(unseen_accs)) * 100
        print(f"\nGeneralization gap: {gap:.1f} points")
        if gap > 10:
            print("Gap is large -- consider more speakers, more augmentation "
                  "variety, or checking that unseen speakers' recording "
                  "conditions (mic, room) aren't just systematically different.")
        else:
            print("Gap is modest -- model is generalizing reasonably to new voices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, required=True)
    parser.add_argument("--speakers_dir", type=str, required=True,
                         help="The ORIGINAL (pre-augmentation, pre-split) recordings dir, "
                              "e.g. ./data/speakers -- so each speaker's raw clips are evaluated once")
    parser.add_argument("--held_out_speakers", nargs="+", default=[])
    args = parser.parse_args()
    main(args)
