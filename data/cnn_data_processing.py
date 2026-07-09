import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # read csv file
        data = pd.read_csv(csv_file, header=None)
        # The first column represent labels and the remaining columns are features
        self.labels = data.iloc[:, 0].values.astype(float)
        self.features = data.iloc[:, 1:].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


def create_dataloaders(train_csv_path, test_csv_path, batch_size=32, transform=None):
    # Load the train and test dataset
    train_dataset = CSVDataset(train_csv_path, transform=transform)
    test_dataset = CSVDataset(test_csv_path, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class ThetaCNNCSVDataset(Dataset):
    """
    Dataset for the raw theta_cnn CSV format written by generate_amg_data.cpp
    (UnifiedAMGDataGenerator::write_sample_theta_cnn):

        n, rho, h, theta, V_00, V_01, ..., V_49_49     (2504 columns)

    where V is the 50x50 std-normalized pooled matrix. Column 0 (n) is not a
    feature or a label -- it's the original matrix size and is dropped here.

    Produces CNNModel-ready samples: X = [theta, -log2(h), V_flat] (2502,), y = rho
    """

    def __init__(self, csv_path, row_indices=None, transform=None):
        data = pd.read_csv(csv_path, header=None, skiprows=_skiprows_for(row_indices))

        rho = data.iloc[:, 1].to_numpy(dtype=float)
        h = data.iloc[:, 2].to_numpy(dtype=float)
        theta = data.iloc[:, 3].to_numpy(dtype=float)
        matrix = data.iloc[:, 4:].to_numpy(dtype=float)

        log_h = -np.log2(h)
        self.features = np.concatenate([theta[:, None], log_h[:, None], matrix], axis=1)
        self.labels = rho
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


def _skiprows_for(row_indices):
    """Build a pandas skiprows callable that keeps only the given 0-based row indices."""
    if row_indices is None:
        return None
    keep = set(int(i) for i in row_indices)
    return lambda row_num: row_num not in keep


def split_csv_row_indices(csv_path, train_ratio=0.8, seed=42, split_policy='sample'):
    """
    Deterministically split an unsplit theta_cnn CSV into train/test row indices.

    Parameters:
        split_policy: 'sample' does a plain shuffled row-level split.
            'h_theta' groups rows sharing the same (h, theta) pair before
            splitting, so no (h, theta) combination straddles train and test
            (this CSV format carries no epsilon/pattern_id/refinement columns,
            so grouping can't go any finer than that).
    """
    meta = pd.read_csv(csv_path, header=None, usecols=[2, 3])
    n_rows = len(meta)
    rng = np.random.default_rng(seed)

    if split_policy == 'sample':
        order = rng.permutation(n_rows)
        split_at = int(round(n_rows * train_ratio))
        train_idx = np.sort(order[:split_at])
        test_idx = np.sort(order[split_at:])
        return train_idx, test_idx

    if split_policy == 'h_theta':
        keys = list(zip(meta[2].round(12), meta[3].round(12)))
        groups = {}
        for row, key in enumerate(keys):
            groups.setdefault(key, []).append(row)

        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        split_at = max(1, min(len(group_keys) - 1, int(round(len(group_keys) * train_ratio))))
        train_keys = set(group_keys[:split_at])

        train_idx, test_idx = [], []
        for key, rows in groups.items():
            (train_idx if key in train_keys else test_idx).extend(rows)
        return np.sort(train_idx), np.sort(test_idx)

    raise ValueError(f"Unknown split_policy '{split_policy}'. Use 'sample' or 'h_theta'.")


def create_theta_cnn_dataloaders_from_csv(csv_path, batch_size=64, num_workers=4,
                                           train_ratio=0.8, seed=42, split_policy='sample'):
    """Build train/test DataLoaders by splitting a single unsplit theta_cnn CSV in-loader."""
    train_idx, test_idx = split_csv_row_indices(
        csv_path, train_ratio=train_ratio, seed=seed, split_policy=split_policy
    )
    print(
        f"Split {len(train_idx) + len(test_idx)} rows from {csv_path} "
        f"using policy='{split_policy}': {len(train_idx)} train / {len(test_idx)} test"
    )

    train_dataset = ThetaCNNCSVDataset(csv_path, row_indices=train_idx)
    test_dataset = ThetaCNNCSVDataset(csv_path, row_indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_csv_file = "datasets/train/raw/train1_cnn.csv"
    test_csv_file = "datasets/test/raw/test1_cnn.csv"
    train_loader, test_loader = create_dataloaders(train_csv_file, test_csv_file, batch_size=8)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs={inputs.shape}, targets={targets.shape}")
        if batch_idx == 0:
            break
