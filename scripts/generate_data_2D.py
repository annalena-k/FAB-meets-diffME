import argparse
import time
from datetime import timedelta

from FABdiffME.utils.generate_samples_2D import generate_samples_2D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Data")
    parser.add_argument(
        "--mename", type=str, default="ee_to_mumu", help="name of matrix element"
    )
    parser.add_argument("--datatype", type=str, default="train", help="train, val, test")
    parser.add_argument("--beginfile", type=str, default=0, help="Number of first file")
    args, _ = parser.parse_known_args()

    # Set parameters for data to be generated
    n_samples = 10000  # 100,000 for training, 10,000 for val and test
    n_samples_per_file = 1000

    name_me = args.mename  # "pi1800" "lambdac" "ee_to_mumu"
    data_type = args.datatype  # "train" "val" "test"
    begin_file = int(args.beginfile)
    print(name_me, data_type, begin_file)

    data_folder = f"data/{name_me}"
    # Set parameters of matrix elements
    dim = 2
    center_of_mass_energy = 1000  # [GeV]
    model_parameters = {}
    # Set vegas parameters
    n_itn = 10
    n_eval1 = 5e4
    n_eval2 = 4e4

    start_time = time.time()
    # Generate samples
    generate_samples_2D(
        n_samples,
        n_samples_per_file=n_samples_per_file,
        begin_file=begin_file,
        name=name_me,
        data_type=data_type,
        data_folder=data_folder,
        dim=dim,
        use_vegas=True,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
        n_itn=n_itn,
        n_eval1=n_eval1,
        n_eval2=n_eval2,
    )
    end_time = time.time()
    print("time [s]:", timedelta(seconds=end_time - start_time))
