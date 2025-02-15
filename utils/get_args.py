"""Define variables and hyperparameters using argparse"""

import argparse
import torch


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device


def get_args(verbose=True):
    """Call args"""
    parser = argparse.ArgumentParser()

    ### Adjustable parameters

    ### WandB and Logging parameters
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--sf-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--op-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--hc-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--oc-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--ppo-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--sac-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )

    ### Environmental / Running parameters
    parser.add_argument(
        "--env-name",
        type=str,
        default="CtF",
        help="This specifies which environment one is working with= FourRooms or CtF1v1, CtF1v2}",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="SNAC",
        help="SNAC / OptionCritic / SAC / PPO",
    )
    parser.add_argument(
        "--grid-type",
        type=int,
        default=0,
        help="0 or 1. Seed to fix the grid, agent, and goal locations",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=None,
        help="episodic length; useful when one wants to constrain to long to short horizon",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1,
        help="Changing this requires redesign of CNN. tensor image size",
    )
    parser.add_argument(
        "--img-tile-size",
        type=int,
        default=32,
        help="32 is default. This is used for logging the images of training progresses. image tile size",
    )
    parser.add_argument(
        "--cost-scaler",
        type=float,
        default=1e-2,
        help="reward shaping parameter r = reawrd - scaler * cost",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )
    parser.add_argument(
        "--post-process",
        type=str,
        default=None,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )

    ### Algorithmic iterations
    parser.add_argument(
        "--SF-epoch",
        type=int,
        default=None,  # 1000
        help="total number of epochs for SFs training",
    )
    parser.add_argument(
        "--OP-timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for OP training",
    )
    parser.add_argument(
        "--HC-timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for HC training",
    )
    parser.add_argument(
        "--OC-timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for OC training",
    )
    parser.add_argument(
        "--PPO-timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for OC training",
    )
    parser.add_argument(
        "--SAC-timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for SAC training",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=None,  # 10
        help="number of iterations within one epoch",
    )

    ### Learning rates
    parser.add_argument(
        "--sf-lr",
        type=float,
        default=None,
        help="SFs train lr where scheduler is used so can be high",
    )
    parser.add_argument(
        "--op-policy-lr", type=float, default=None, help="Option network lr"
    )
    parser.add_argument(
        "--op-critic-lr",
        type=float,
        default=None,
        help="Option policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--hc-policy-lr",
        type=float,
        default=None,
        help="Hierarchical Controller network lr",
    )
    parser.add_argument(
        "--hc-critic-lr",
        type=float,
        default=None,
        help="Hierarchical Policy policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "-oc-policy-lr",
        type=float,
        default=None,
        help="Hierarchical Policy policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--oc-critic-lr",
        type=float,
        default=None,
        help="Hierarchical Policy policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--ppo-policy-lr", type=float, default=None, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--ppo-critic-lr",
        type=float,
        default=None,
        help="PPO-critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--sac-policy-lr", type=float, default=None, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--sac-critic-lr",
        type=float,
        default=None,
        help="PPO-critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--sac-alpha-lr",
        type=float,
        default=1e-4,
        help="Lr for auto-tune entropy scaler",
    )

    ### Algorithmic parameters
    parser.add_argument(
        "--PM-policy",
        type=str,
        default="RW",
        help="PPO policy entropy scaler",
    )
    parser.add_argument("--gamma", type=float, default=None, help="discount parameters")
    parser.add_argument(
        "--min-option-length",
        type=int,
        default=10,
        help="Minimum time step for one option duration of SNAC / EigenOption",
    )
    parser.add_argument(
        "--warm-batch-size",
        type=int,
        default=None,
        help="Base number of batch size for training",
    )
    parser.add_argument(
        "--DIF-batch-size",
        type=int,
        default=None,
        help="Base number of batch size for training",
    )
    parser.add_argument(
        "--op-mode",
        type=str,
        default=None,
        help="Base number of batch size for training",
    )
    parser.add_argument(
        "--sf-batch-size",
        type=int,
        default=None,
        help="Base number of batch size for training",
    )
    parser.add_argument(
        "--op-batch-size",
        type=int,
        default=None,
        help="Option policy number of batch size for training",
    )
    parser.add_argument(
        "--hc-batch-size",
        type=int,
        default=None,
        help="Hierarchical policy number of batch size for training",
    )
    parser.add_argument(
        "--oc-batch-size",
        type=int,
        default=None,
        help="Option critic number of batch size for training",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=None,
        help="Naive ppo number of batch size for training",
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=None,
        help="SAC number of batch size for training",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="SAC number of batch size for training",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=None,
        help="SAC number of batch size for training",
    )
    parser.add_argument(
        "--min-batch-for-worker",
        type=int,
        default=2048,
        help="Minimum batch size assgined for one worker (thread)",
    )
    parser.add_argument(
        "--op-min-batch-for-worker",
        type=int,
        default=10240,
        help="Minimum batch size assgined for one worker (thread)",
    )
    parser.add_argument(
        "--op-entropy-scaler",
        type=float,
        default=1e-3,
        help="Option policy entropy scaler",
    )
    parser.add_argument(
        "--hc-entropy-scaler",
        type=float,
        default=3e-3,
        help="Hierarchical policy entropy scaler",
    )
    parser.add_argument(
        "--sac-entropy-scaler",
        type=float,
        default=1e-3,
        help="PPO policy entropy scaler",
    )
    parser.add_argument(
        "--ppo-entropy-scaler",
        type=float,
        default=1e-3,
        help="PPO policy entropy scaler",
    )

    ### SF param (loss scale)
    parser.add_argument(
        "--reward-loss-scaler",
        type=float,
        default=None,
        help="Scaler to SFs reward regression loss",
    )
    parser.add_argument(
        "--state-loss-scaler",
        type=float,
        default=None,
        help="Scaler to SFs latent state regression loss",
    )
    parser.add_argument(
        "--weight-loss-scaler",
        type=float,
        default=None,
        help="Scaler to SFs latent state regression loss (VAE only)",
    )
    parser.add_argument(
        "--lasso-loss-scaler",
        type=float,
        default=None,
        help="Scaler to SFs network weight to prevent overfitting",
    )
    parser.add_argument(
        "--max-num-traj",
        type=int,
        default=None,
        help="Maximum number of trajectories the buffer can store. Exceeding it will refresh the oldest trajectory",
    )
    parser.add_argument(
        "--min-num-traj",
        type=int,
        default=None,
        help="Minimum number of trajectory to start training.",
    )

    ### Resorces
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of threads to use in sampling. If none, sampler will select available threads number with this limit",
    )
    parser.add_argument(
        "--cpu-preserve-rate",
        type=float,
        default=0.95,
        help="For multiple run of experiments, one can set this to restrict the cpu threads the one exp uses for sampling.",
    )

    ### Dimensional params
    parser.add_argument(
        "--a-dim",
        type=int,
        default=None,
        help="action dimension. For grid with 5 available actions, it is one-hotted to be 1 x 5.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--snac-split-ratio",
        type=float,
        default=0.5,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--temporal-balance-ratio",
        type=float,
        default=0.25,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=None,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--sf-dim",
        type=int,
        default=None,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--sf-fc-dim",
        type=int,
        default=None,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--op-policy-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--op-critic-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--hc-policy-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--hc-critic-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--oc-fc-dim",
        type=int,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--oc-termination-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--oc-critic-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--ppo-policy-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--ppo-critic-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--sac-policy-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--sac-critic-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )

    # PPO parameters
    parser.add_argument(
        "--K-epochs", type=int, default=None, help="PPO update per one iter"
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="clipping parameter for gradient"
    )
    parser.add_argument(
        "--target-kl", type=float, default=None, help="clipping parameter for gradient"
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Used in advantage estimation.",
    )

    # SAC parameters
    parser.add_argument(
        "--tune-alpha", type=bool, default=True, help="Automatic entropy scaler."
    )
    parser.add_argument(
        "--sac-init-alpha",
        type=float,
        default=0.2,
        help="Initial entropy scaler",
    )
    parser.add_argument(
        "--sac-soft-update-rate",
        type=float,
        default=0.005,
        help="Target critic network update. Lower the slower rate of update",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="Interval to perform target critic update in SAC",
    )

    # Misc. parameters
    parser.add_argument(
        "--rendering",
        type=bool,
        default=True,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=None,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--draw-map",
        type=bool,
        default=True,
        help="Turn off plotting reward map. Only works for FourRoom",
    )
    parser.add_argument(
        "--import-sf-model",
        action="store_true",
        help="Imports previously trained SF model",
    )
    parser.add_argument(
        "--import-op-model",
        action="store_true",
        help="Imports previously trained OP model",
    )
    parser.add_argument(
        "--import-hc-model",
        action="store_true",
        help="Imports previously trained HC model",
    )
    parser.add_argument(
        "--import-ppo-model",
        action="store_true",
        help="Imports previously trained PPO model",
    )
    parser.add_argument(
        "--import-sac-model",
        action="store_true",
        help="Imports previously trained SAC model",
    )
    parser.add_argument(
        "--import-oc-model",
        action="store_true",
        help="Imports previously trained OC model",
    )

    parser.add_argument("--gpu-idx", type=int, default=0, help="gpu idx to train")
    parser.add_argument("--verbose", type=bool, default=False, help="WandB logging")

    args = parser.parse_args()

    # post args processing
    args.device = select_device(args.gpu_idx, verbose)

    if args.import_op_model and not args.import_sf_model:
        print("\tWarning: importing OP model without Pre-trained SF")
    if (args.import_hc_model and not args.import_op_model) or (
        args.import_hc_model and not args.import_sf_model
    ):
        print("\tWarning: importing HC model without Pre-trained SF/OP")

    return args
