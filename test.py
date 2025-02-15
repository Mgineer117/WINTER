import uuid
import random
import wandb
import sys
import os
import contextlib
from tqdm import tqdm  # Progress bar

from algorithms import SNAC, EigenOption, PPO, SAC, OptionCritic
from utils.call_env import call_env
from utils.utils import setup_logger, seed_all, override_args

wandb.require("core")
wandb.init(mode="disabled")  # Turn off WandB sync


#########################################################
# Utility to Suppress Print Output and Capture Errors
#########################################################
@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull  # Redirect output
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr  # Restore output


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id):
    """Initiate the training process with given args."""
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

    envs = ["FourRooms", "Maze", "CtF", "PointNavigation"]
    errors = []  # Store errors to print later

    with tqdm(
        total=len(envs),
        desc="Testing Any Environmental Failures",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        for env_name in envs:
            try:
                args = override_args(env_name)
                args.algo_name = "PPO"
                args.PPO_timesteps = 10000  # Small timesteps for quick testing
                args.min_batch_for_worker = 1024
                args.rendering = True
                args.draw_map = True

                with suppress_output():  # Suppress print output
                    train(args, seed, unique_id)

            except Exception as e:
                errors.append(f"❌ PPO Test Failed on {env_name}: {str(e)}")

            pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)


#########################################################
# TEST CASE 2: Run SNAC with small SF-epoch and OP/HC-timesteps
#########################################################
def test_snac():
    unique_id = str(uuid.uuid4())[:4]
    seed = 42
    seed_all(seed)
    errors = []  # Store errors to print later

    with tqdm(
        total=1,
        desc="Testing SNAC Pipeline",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        try:
            args = override_args("CtF")
            args.algo_name = "SNAC"
            args.SF_epoch = 10  # Small SF-epoch for quick testing
            args.step_per_epoch = 1  # Small steps per epoch
            args.OP_timesteps = 10000  # Small OP-timesteps
            args.HC_timesteps = 10000  # Small HC-timesteps
            args.min_batch_size = 2048
            args.max_batch_size = 4096
            args.warm_batch_size = 2048
            args.DIF_batch_size = 4096
            args.post_process = None
            args.num_options = 2
            args.method = "top"
            args.rendering = True
            args.draw_map = True

            with suppress_output():  # Suppress print output
                train(args, seed, unique_id)

        except Exception as e:
            errors.append(f"❌ SNAC Test Failed: {str(e)}")

        pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)


#########################################################
# TEST CASE 3: Run EigenOption with small SF-epoch and OP/HC-timesteps
#########################################################
def test_eigenoption():
    seed = 42
    seed_all(seed)
    unique_id = str(uuid.uuid4())[:4]
    errors = []  # Store errors to print later

    with tqdm(
        total=1,
        desc="Testing EigenOption pipeline",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        try:
            args = override_args("PointNavigation")
            args.algo_name = "EigenOption"
            args.SF_epoch = 10  # Small SF-epoch for quick testing
            args.step_per_epoch = 1  # Small steps per epoch
            args.OP_timesteps = 10000  # Small OP-timesteps
            args.HC_timesteps = 10000  # Small HC-timesteps
            args.min_batch_size = 2048
            args.max_batch_size = 4096
            args.warm_batch_size = 2048
            args.DIF_batch_size = 4096
            args.post_process = None
            args.num_options = 2
            args.method = "top"
            args.rendering = True
            args.draw_map = True

            with suppress_output():  # Suppress print output
                train(args, seed, unique_id)

        except Exception as e:
            errors.append(f"❌ EigenOption Test Failed: {str(e)}")

        pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)


#########################################################
# Run Tests
#########################################################
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🚀 Running PPO for small timesteps on all environments ")
    print("=" * 60 + "\n")
    test_ppo()

    print("\n" + "=" * 60)
    print(" 🎯 Running SNAC pipeline with small SF-epoch, OP, HC-timesteps ")
    print("=" * 60 + "\n")
    test_snac()

    print("\n" + "=" * 60)
    print(" 🧠 Running EigenOption pipeline with small SF-epoch, OP, HC-timesteps ")
    print("=" * 60 + "\n")
    test_eigenoption()

    print("\n✅ All tests completed successfully!")
