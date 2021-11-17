import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import SequentialDistributedSampler


class IMDB(Dataset):
    def __init__(self, tok, text, label):
        self.tok = tok
        self.text = text
        self.label = label

        assert len(self.text) == len(self.label)
        print(f"Load {len(self.label)} data.")

    def __getitem__(self, idx):
        src = self.tok(
            self.text[idx], truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"],
            "token_type_ids": src["token_type_ids"],
            "attention_mask": src["attention_mask"],
            "labels": torch.tensor(self.label[idx]),
        }

    def __len__(self):
        return len(self.label)


def get_trn_dev_loader(dset, tok, batch_size, workers, distributed=False) -> DataLoader:
    """
    Return:
        Tuple[DataLoader]
    """
    # trn 20000, dev 5000
    trn_text, dev_text, trn_label, dev_label = train_test_split(
        dset["text"], dset["label"], test_size=0.2
    )
    trn_dset = IMDB(tok, trn_text, trn_label)
    dev_dset = IMDB(tok, dev_text, dev_label)

    shuffle_flag = True
    trn_sampler, dev_sampler = None, None
    if distributed:
        trn_sampler = DistributedSampler(trn_dset)
        dev_sampler = SequentialDistributedSampler(dev_dset)
        shuffle_flag = False

    trn_loader = DataLoader(
        dataset=trn_dset,
        batch_size=batch_size,
        sampler=trn_sampler,
        shuffle=shuffle_flag,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dataset=dev_dset,
        batch_size=batch_size,
        sampler=dev_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return (trn_loader, dev_loader)


def get_tst_loader(dset, tok, batch_size, workers, distributed=False) -> DataLoader:
    """
    Return:
        DataLoader
    """
    tst_dset = IMDB(tok, dset["text"], dset["label"])

    tst_sampler = None
    if distributed:
        tst_sampler = SequentialDistributedSampler(tst_dset)

    return DataLoader(
        dataset=tst_dset,
        batch_size=batch_size,
        sampler=tst_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
