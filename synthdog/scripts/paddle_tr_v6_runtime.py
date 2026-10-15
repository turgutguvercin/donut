"""Strict, width-bucketed loader using the actual pinned PaddleOCR operators."""
from collections import Counter, defaultdict
import copy
import logging
from pathlib import Path, PurePosixPath

import numpy as np
import paddle

from paddle_v6_utils import WIDTHS, MAX_LENGTH, bucket_batches, characters


class TurkishBucketDataset(paddle.io.Dataset):
    def __init__(self, config, mode, logger=None):
        super().__init__()
        from ppocr.data.imaug import create_operators
        self.need_reset = False
        self.root = Path(config[mode]["dataset"]["data_dir"])
        self.characters = characters(config["Global"]["character_dict_path"])
        self.ctc_chars = ["blank"] + self.characters
        self.nrtr_chars = ["blank", "<unk>", "<s>", "</s>"] + self.characters
        self.records, self.groups, self.ops = [], defaultdict(list), {}
        paths = config[mode]["dataset"]["label_file_list"]
        if len(paths) != 1:
            raise ValueError("Use one full/recommended/challenge split list")
        main_list = Path(paths[0])
        expected = main_list.read_text(encoding="utf-8").splitlines()
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate label rows")
        seen = []
        for width in config["TurkishWidthBuckets"]["widths"]:
            if width not in WIDTHS:
                raise ValueError("Unvalidated width bucket")
            transforms = copy.deepcopy(config[mode]["dataset"]["transforms"])
            transforms[2]["RecResizeImg"]["image_shape"] = [3, 48, width]
            self.ops[width] = create_operators(transforms, config["Global"])
            path = main_list.with_name(main_list.stem + f"-w{width}.txt")
            for record in path.read_text(encoding="utf-8").splitlines():
                name, label = record.split("\t", 1)
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                    raise ValueError("Unsafe image path")
                if not label or len(label) > MAX_LENGTH - 2 or set(label) - set(self.characters):
                    raise ValueError("Label exceeds model capacity or dictionary")
                required = len(label) + sum(a == b for a, b in zip(label, label[1:]))
                if required > width // 8:
                    raise ValueError("Insufficient measured v6 CTC capacity")
                self.groups[width].append(len(self.records))
                self.records.append((name, label, width))
                seen.append(record)
        if Counter(seen) != Counter(expected) or len({r[0] for r in self.records}) != len(self.records):
            raise ValueError("Width lists must partition the complete selected split exactly")
        (logger or logging.getLogger(__name__)).info("Turkish %s loader: %s samples across %s widths", mode, len(self), len(self.groups))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, label, width = self.records[index]
        ops = self.ops[width]
        data = ops[0]({"image": (self.root / name).read_bytes(), "label": label})
        if data is None or data["image"].shape[0] != 48 or data["image"].shape[1] > width:
            raise ValueError(f"Invalid image or horizontal squeezing: {name}")
        actual_width = data["image"].shape[1]
        data = ops[1](data)
        if data is None or int(data["length"]) != len(label):
            raise ValueError(f"Rejected or truncated label: {name}")
        decoded = "".join(self.ctc_chars[int(i)] for i in data["label_ctc"][:len(label)])
        nrtr = data["label_gtc"]
        if (decoded != label or int(nrtr[0]) != 2 or int(nrtr[len(label) + 1]) != 3
                or "".join(self.nrtr_chars[int(i)] for i in nrtr[1:len(label) + 1]) != label
                or np.any(nrtr[len(label) + 2:] != 0)):
            raise ValueError(f"Label encoding round trip failed: {name}")
        data = ops[2](data)
        resized_width = round(float(data["valid_ratio"]) * width)
        if (data["image"].shape != (3, 48, width) or not actual_width <= resized_width <= actual_width + 1
                or not np.isfinite(data["image"]).all() or np.any(data["image"][:, :, resized_width:] != 0)):
            raise ValueError(f"Invalid normalization or padding: {name}")
        return ops[3](data)


class TurkishBucketSampler(paddle.io.Sampler):
    def __init__(self, dataset, batch_sizes, shuffle, seed):
        super().__init__()
        self.groups, self.batch_sizes = dataset.groups, batch_sizes
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __iter__(self):
        result = bucket_batches(self.groups, self.batch_sizes, self.seed, self.epoch, self.shuffle)
        self.epoch += 1
        yield from result

    def __len__(self):
        return sum((len(indices) + self.batch_sizes[w] - 1) // self.batch_sizes[w] for w, indices in self.groups.items())

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_dataloader(config, mode, device, logger, seed=None):
    if not config.get("TurkishWidthBuckets", {}).get("enabled"):
        from ppocr.data import build_dataloader as upstream
        return upstream(config, mode, device, logger, seed)
    if config["Global"].get("distributed") or paddle.distributed.get_world_size() != 1:
        raise ValueError("This bucket loader supports one CPU/GPU process; distributed training needs a rank-aware sampler")
    dataset = TurkishBucketDataset(config, mode, logger)
    options = config["TurkishWidthBuckets"]
    sampler = TurkishBucketSampler(dataset, {int(k): v for k, v in options["batch_sizes"].items()}, mode == "Train", options["seed"])
    loader = config[mode]["loader"]
    return paddle.io.DataLoader(dataset, batch_sampler=sampler, places=device,
        num_workers=loader["num_workers"], return_list=True,
        use_shared_memory=loader.get("use_shared_memory", True))
