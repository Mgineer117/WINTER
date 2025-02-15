import uuid
import random
from algorithms import SNAC, EigenOption, PPO, SAC, OptionCritic

from utils.call_env import call_env
from utils.utils import setup_logger, seed_all, override_args


import wandb

wandb.require("core")


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id):
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
    logger, writer = setup_logger(args, unique_id, seed)

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

if __name__ == "__main__":
    init_args = override_args()
    unique_id = str(uuid.uuid4())[:4]

    seed_all(init_args.seed)
    seeds = [random.randint(1, 100_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args()
        train(args, seed, unique_id)
