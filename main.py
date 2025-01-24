import uuid
import random
from algorithms import FeatureTrain, WINTER, PPO, SAC, OptionCritic

from utils.call_env import call_env
from utils.get_args import get_args
from utils.call_feature_weights import call_feature_weights
from utils.utils import setup_logger, seed_all, load_hyperparams


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

    if args.algo_name == "PPO":
        alg = PPO(
            env=env,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "SAC":
        alg = SAC(
            env=env,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "OptionCritic":
        alg = OptionCritic(
            env=env,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "WINTER":
        reward_feature_weights, reward_options, state_options = call_feature_weights(
            args.sf_r_dim, args.sf_s_dim
        )
        ft = FeatureTrain(
            env=env,
            logger=logger,
            writer=writer,
            reward_feature_weights=reward_feature_weights,
            args=args,
        )
        sf_network, prev_epoch = ft.train()
        alg = WINTER(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            reward_options=reward_options,
            state_options=state_options,
            args=args,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# ENV LOOP
#########################################################


def override_args():
    args = get_args(verbose=False)
    file_path = "assets/env_params.json"
    current_params = load_hyperparams(file_path=file_path, env_name=args.env_name)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


if __name__ == "__main__":
    args = override_args()
    unique_id = str(uuid.uuid4())[:4]

    seed_all(args.seed)
    seeds = [random.randint(1, 100_000) for _ in range(args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        train(args, seed, unique_id)
