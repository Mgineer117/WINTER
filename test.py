import uuid
import random
import wandb

from algorithms import SNAC, EigenOption, PPO, SAC, OptionCritic
from utils.call_env import call_env
from utils.utils import setup_logger, seed_all, override_args

wandb.require("core")
wandb.init(mode="disabled")  # Turn off WandB sync


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id):
    """Initiate the training process upon given args.

    Args:
        args (arguments): includes all hyperparameters.
        unique_id (int): Unique running ID for the experiment.
    """
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
# TEST CASE 1: Run PPO for small timesteps on all environments
#########################################################
def test_ppo():
    seed = 42
    seed_all(seed)
    unique_id = str(uuid.uuid4())[:4]

    for env_name in ["FourRooms", "Maze", "CtF", "PointNavigation"]:
        print(f"Testing PPO on {env_name} with small timesteps...")
        args = override_args(env_name)
        args.algo_name = "PPO"
        args.PPO_timesteps = 10000  # Small number of timesteps for quick testing
        args.min_batch_for_worker = 1024
        train(args, seed, unique_id)


#########################################################
# TEST CASE 2: Run SNAC with small SF-epoch and OP/HC-timesteps
#########################################################
def test_snac():
    unique_id = str(uuid.uuid4())[:4]
    seed = 42
    seed_all(seed)

    print(f"Testing SNAC pipeline on FourRooms with small epochs and timesteps...")
    env_name = "CtF"
    args = override_args(env_name)
    args.algo_name = "SNAC"
    args.SF_epoch = 10  # Small SF-epoch for quick testing
    args.step_per_epoch = 1  # Small steps per epoch
    args.OP_timesteps = 10000  # Small OP-timesteps
    args.HC_timesteps = 10000  # Small HC-timesteps
    train(args, seed, unique_id)


#########################################################
# Run Tests
#########################################################
if __name__ == "__main__":
    print("-------------------------------------------------------")
    print(" Running PPO for small timesteps on all environments ")
    print("-------------------------------------------------------")
    test_ppo()

    print("\n-------------------------------------------------------")
    print(" Running SNAC pipeline with small SF-epoch, OP, HC-timesteps ")
    print("-------------------------------------------------------")
    test_snac()
