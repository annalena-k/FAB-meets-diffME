import gzip
import pickle
from pathlib import Path

from FABdiffME.targets.target_util import load_target, read_madgraph_phasespace_points

# Preprocess batch of .lhe.gz files and save samples to .pt file for faster loading during training
# Samples are saved to the same folder/filename file.pt as the input

if __name__ == "__main__":
    data_path = "data"
    data_type = "train"
    no_files = 10
    n_samples_per_file = int(1e6)

    target_name = "ee_to_ttbar_wb"
    dim = 8
    center_of_mass_energy = 1000  # [GeV]

    target = load_target(
        name=target_name,
        dim=dim,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters={},
        epsilon_boundary=1.0e-8,
    )

    for n in range(no_files):
        filename_in = Path(
            f"{data_path}/{target_name}/unweighted_events_{data_type}_{n}.lhe.gz"
        )
        if filename_in.is_file():
            print(f"Loading {filename_in}")
            momenta = read_madgraph_phasespace_points(
                filename_in, target, n_samples_per_file
            )
            filename_out = (
                f"{data_path}/{target_name}/unweighted_events_{data_type}_{n}.pt"
            )
            with gzip.open(filename_out, "wb") as f:
                pickle.dump(momenta, f)
            print(filename_in, "with", len(momenta), "saved to", filename_out)

            print("Loading it again ...")
            with gzip.open(filename_out, "rb") as f:
                moms = pickle.load(f)
            print("Loaded momenta of shape", moms.shape)
        else:
            print(f"File {filename_in} does not exist.")
