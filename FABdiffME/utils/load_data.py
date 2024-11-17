from typing import Optional
import chex
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset

from FABdiffME.targets.target_util import (
    Target,
    read_madgraph_phasespace_points,
    read_preprocssed_phasespace_points,
)


class CustomDataset(Dataset):
    def __init__(self, data: chex.Array):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> chex.Array:
        return self.data[idx]


def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def load_data(
    path_to_data: str,
    data_type: str,
    generator: str,
    target: Optional[Target],
    n_data_samples: int,
    n_samples_per_file: int,
):

    n_files = int(jnp.ceil(n_data_samples / n_samples_per_file))
    data = jnp.zeros([n_data_samples, target.dim])
    count_samples = 0
    for n in range(n_files):
        # Determine number of samples that have to be loaded
        if n_data_samples - count_samples >= n_samples_per_file:
            n_samples_load = n_samples_per_file
        else:
            n_samples_load = n_data_samples - count_samples
        # Load samples dependent on type of generator
        if generator == "madgraph":
            filename = f"{path_to_data}/unweighted_events_{data_type}_{n}.lhe.gz"
            data_subset = read_madgraph_phasespace_points(
                filename, target, n_samples_load
            )
        elif generator == "madgraph_preprocessed":
            filename = f"{path_to_data}/unweighted_events_{data_type}_{n}.pt"
            data_subset = read_preprocssed_phasespace_points(filename, n_samples_load)
        elif generator == "vegas":
            filename = (
                f"{path_to_data}/{data_type}_data_vegas_{n_samples_per_file}_{n}.npy"
            )
            with open(filename, "rb") as f:
                data_subset = jnp.load(f)
        elif generator == "rej":
            filename = (
                f"{path_to_data}/{data_type}_data_unit_{n_samples_per_file}_{n}.npy"
            )
            with open(filename, "rb") as f:
                data_subset = jnp.load(f)
        else:
            data_subset = None
            assert (
                True
            ), f"Generator {generator} not in ['madgraph', 'madgraph_preprocessed', 'vegas', 'rej']"

        assert (
            data_subset.shape[-1] == target.dim
        ), f"Data read from Madgraph file {filename} does not have required shape [:,{target.dim}], but has {data_subset.shape}"
        # Add loaded samples to data array
        n_subset = data_subset.shape[0]
        data = data.at[count_samples : count_samples + n_subset, :].set(data_subset)
        count_samples += n_subset

    return data
