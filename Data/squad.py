import os

import numpy as np
import torch
from torch.utils.data import Dataset


def require_file(path: str, hint: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")


def sanity_check_cache(args):
    require_file(args.train_npz, "Run preproc.py and place train.npz under data/")
    require_file(args.dev_npz, "Run preproc.py and place dev.npz under data/")
    require_file(args.word_emb_json, "Run preproc.py to generate word_emb.json")
    require_file(args.char_emb_json, "Run preproc.py to generate char_emb.json")
    require_file(args.train_eval_json, "Run preproc.py to generate train_eval.json")
    require_file(args.dev_eval_json, "Run preproc.py to generate dev_eval.json")

    data = np.load(args.train_npz)
    for k in ["context_idxs", "context_char_idxs", "ques_idxs", "ques_char_idxs", "y1s", "y2s", "ids"]:
        if k not in data:
            raise KeyError(f"{args.train_npz} missing key: {k}")
    if data["ids"].shape[0] <= 0:
        raise ValueError("train.npz has zero samples.")
    if (data["y1s"] > data["y2s"]).any():
        raise ValueError("Found y1 > y2 in train.npz (span start > end).")


class SQuADDataset(Dataset):
    def __init__(self, npz_file: str):
        super().__init__()
        data = np.load(npz_file)

        self.context_idxs      = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        self.ques_idxs         = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs    = torch.from_numpy(data["ques_char_idxs"]).long()
        self.y1s               = torch.from_numpy(data["y1s"]).long()
        self.y2s               = torch.from_numpy(data["y2s"]).long()
        self.ids               = torch.from_numpy(data["ids"]).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return (
            self.context_idxs[idx],
            self.context_char_idxs[idx],
            self.ques_idxs[idx],
            self.ques_char_idxs[idx],
            self.y1s[idx],
            self.y2s[idx],
            self.ids[idx],
        )
