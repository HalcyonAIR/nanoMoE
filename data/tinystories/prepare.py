#!/usr/bin/env python3
"""
Prepare TinyStories dataset for domain-shift experiment.

TinyStories is synthetic children's stories - very different from Shakespeare.
Good for testing domain shift without massive download.

IMPORTANT: Uses Shakespeare's vocabulary to enable fine-tuning without
vocab size mismatch. Characters not in Shakespeare vocab are filtered out.
"""

import os
import pickle
import numpy as np
from datasets import load_dataset

def prepare_tinystories():
    print("Loading TinyStories dataset...")
    # Load a small subset for quick experiments
    dataset = load_dataset("roneneldan/TinyStories", split="train[:50000]")

    # Load Shakespeare vocab to ensure compatibility for fine-tuning
    shakespeare_meta_path = os.path.join(
        os.path.dirname(__file__), '..', 'shakespeare_char', 'meta.pkl'
    )

    if os.path.exists(shakespeare_meta_path):
        print(f"Loading Shakespeare vocab from {shakespeare_meta_path}")
        with open(shakespeare_meta_path, 'rb') as f:
            sh_meta = pickle.load(f)
        stoi = sh_meta['stoi']
        itos = sh_meta['itos']
        vocab_size = sh_meta['vocab_size']
        valid_chars = set(stoi.keys())
        print(f"Using Shakespeare vocab (size {vocab_size})")
    else:
        print("WARNING: Shakespeare vocab not found, using TinyStories vocab")
        # Fallback to building vocab from TinyStories
        all_text = "\n\n".join(dataset["text"])
        chars = sorted(list(set(all_text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        valid_chars = set(chars)

    # Combine all stories into one text, filtering invalid chars
    print("Combining stories (filtering to Shakespeare vocab)...")
    filtered_texts = []
    chars_removed = 0
    total_chars = 0

    for text in dataset["text"]:
        filtered = ''.join(c for c in text if c in valid_chars)
        chars_removed += len(text) - len(filtered)
        total_chars += len(text)
        if filtered.strip():  # Only add non-empty stories
            filtered_texts.append(filtered)

    all_text = "\n\n".join(filtered_texts)

    print(f"Total characters: {len(all_text):,}")
    print(f"Characters filtered out: {chars_removed:,} ({100*chars_removed/total_chars:.1f}%)")
    print(f"Vocab size: {vocab_size}")

    # Encode
    def encode(s):
        return [stoi[c] for c in s]

    # Train/val split (90/10)
    n = len(all_text)
    train_data = all_text[:int(n * 0.9)]
    val_data = all_text[int(n * 0.9):]

    # Encode to numpy arrays
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Save
    output_dir = os.path.dirname(__file__)
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))

    # Save meta (using Shakespeare's vocab)
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    prepare_tinystories()
