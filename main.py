import uuid
import random
import datetime
from algorithms import SNAC, EigenOption, PPO, SAC, OptionCritic

from utils.call_env import call_env
from utils.utils import setup_logger, seed_all, override_args


import wandb

wandb.require("core")


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id, exp_time):
    """Initiate the training process upon given args

    Args:
        args (arguments): includes all hyperparameters
            - Algorithms: SNAC, EigenOption, CoveringOption, PPO
                - The '+' sign after the algorithm denotes clustering
                    - +: clustering in eigenspace
                    - ++: clustering in value space
        unique_id (int): This is an unique running id for the experiment
    """
    # # call logger
    env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    algo_classes = {
        "PPO": PPO,
        "SAC": SAC,
        "OptionCritic": OptionCritic,
        "SNAC": SNAC,
        "EigenOption": EigenOption,
    }

    alg_class = algo_classes.get(args.algo_name)
    if alg_class is None:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg = alg_class(
        env=env,
        logger=logger,
        writer=writer,
        args=args,
    )
    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# ENV LOOP
#########################################################
import os
import pandas as pd


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")


# Example usage


if __name__ == "__main__":
    init_args = override_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    seed_all(init_args.seed)
    seeds = [random.randint(1, 100_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args()
        train(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
