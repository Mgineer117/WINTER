import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Union


from models.layers.op_networks import OptionPolicy, OptionCritic, OPtionCriticTwin
from models.layers.hc_networks import HC_Policy, HC_PPO, HC_RW, HC_Critic
from models.layers.oc_networks import OC_Policy, OC_Critic
from models.layers.ppo_networks import PPO_Policy, PPO_Critic
from models.layers.sac_networks import SAC_Policy, SAC_CriticTwin

from log.logger_util import colorize


def get_conv_layer(args):
    _, _, in_channels = args.s_dim

    if args.env_name == "OneRoom":
        encoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": in_channels,  # Number of input channels
                "out_filters": 32,
            },  # Maintain spatial size (9x9 -> 9x9)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Reduce spatial size (9x9 -> 5x5)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Maintain spatial size (5x5 -> 5x5)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 128,
                "out_filters": 256,
            },  # Reduce spatial size (5x5 -> 3x3)
        ]

        decoder_conv_layers = [
            {
                "type": "conv_transpose",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 0,
                "activation": nn.ELU(),
                "in_filters": 256,
                "out_filters": 128,
            },  # Increases size: (3x3 -> 6x6)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 128,
                "out_filters": 64,
            },  # Maintains size: (6x6 -> 6x6)
            {
                "type": "conv_transpose",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 0,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 32,
            },  # Increases size: (6x6 -> 9x9)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),  # Final activation for reconstruction
                "in_filters": 32,
                "out_filters": in_channels,  # Number of output channels
            },  # Maintains size: (9x9 -> 9x9)
        ]

    elif args.env_name in ("LavaRooms", "CtF"):
        encoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": in_channels,  # Number of input channels
                "out_filters": 32,
            },  # Maintain spatial size (12x12 -> 12x12)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Reduce spatial size (12x12 -> 6x6)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Maintain spatial size (6x6 -> 6x6)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 128,
                "out_filters": 256,
            },  # Reduce spatial size (6x6 -> 3x3)
        ]

        decoder_conv_layers = [
            {
                "type": "conv_transpose",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
                "activation": nn.ELU(),
                "in_filters": 256,
                "out_filters": 128,
            },  # Increases size: (3x3 -> 5x5)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),
                "in_filters": 128,
                "out_filters": 64,
            },  # Maintains size: (5x5 -> 5x5)
            {
                "type": "conv_transpose",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 32,
            },  # Increases size: (5x5 -> 9x9)
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.ELU(),  # Final activation for reconstruction
                "in_filters": 32,
                "out_filters": in_channels,  # Number of output channels
            },  # Maintains size: (9x9 -> 9x9)
        ]

    return encoder_conv_layers, decoder_conv_layers


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def call_sacNetwork(args):
    from models.policy import SAC_Learner

    if args.import_sac_model:
        print("Loading previous SAC parameters....")
        policy, critic_twin, alpha, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/sac_model.p", "rb")
        )
    else:
        # Define the Actor (Policy) network
        actor = SAC_Policy(
            input_dim=args.flat_s_dim,
            hidden_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.ReLU(),
            is_discrete=args.is_discrete,
        )

        # Define the Critic networks
        critic_twin = SAC_CriticTwin(
            input_dim=args.flat_s_dim + args.a_dim,
            fc_dim=args.fc_dim,
            activation=nn.ReLU(),
        )

        alpha = args.sac_init_alpha

    # Create the SAC Learner
    policy = SAC_Learner(
        policy=actor,
        critic_twin=critic_twin,
        alpha=alpha,
        normalizer=normalizer,
        policy_lr=args.sac_policy_lr,
        critic_lr=args.sac_critic_lr,
        alpha_lr=args.sac_alpha_lr,
        tau=args.sac_soft_update_rate,
        gamma=args.gamma,
        batch_size=args.sac_batch_size,
        target_update_interval=args.target_update_interval,
        tune_alpha=args.tune_alpha,
        device=args.device,
    )

    return policy


def call_ppoNetwork(args):
    from models.policy import PPO_Learner

    if args.import_ppo_model:
        print("Loading previous PPO parameters....")
        policy, critic, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/ppo_model.p", "rb")
        )
    else:
        actor = PPO_Policy(
            input_dim=args.flat_s_dim,
            hidden_dim=args.ppo_policy_dim,
            a_dim=args.a_dim,
            activation=nn.Tanh(),
            is_discrete=args.is_discrete,
        )
        critic = PPO_Critic(
            input_dim=args.flat_s_dim,
            hidden_dim=args.ppo_critic_dim,
            activation=nn.Tanh(),
        )

    policy = PPO_Learner(
        policy=actor,
        critic=critic,
        policy_lr=args.ppo_policy_lr,
        critic_lr=args.ppo_critic_lr,
        minibatch_size=args.ppo_batch_size,
        entropy_scaler=args.ppo_entropy_scaler,
        eps=args.eps_clip,
        gae=args.gae,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_ocNetwork(args):
    from models.policy import OC_Learner

    if args.import_oc_model:
        print("Loading previous OC parameters....")
        policy, critic, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/oc_model.p", "rb")
        )
    else:
        encoder_conv_layers, _ = get_conv_layer(args)
        policy = OC_Policy(
            state_dim=args.s_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            num_options=args.num_vector,
            encoder_conv_layers=encoder_conv_layers,
            activation=nn.Tanh(),
            is_discrete=args.is_discrete,
        )
        critic = OC_Critic(
            input_dim=args.flat_s_dim,
            fc_dim=args.fc_dim,
            num_options=args.num_vector,
            activation=nn.Tanh(),
        )

    policy = OC_Learner(
        policy=policy,
        critic=critic,
        normalizer=normalizer,
        policy_lr=args.ppo_policy_lr,
        critic_lr=args.ppo_critic_lr,
        entropy_scaler=args.ppo_entropy_scaler,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_sfNetwork(args, sf_path: str | None = None):
    from models.layers.sf_networks import AutoEncoder, VAE, ConvNetwork
    from models.policy import SF_LASSO, SF_SNAC, SF_EigenOption

    if args.algo_name == "SNAC":
        snac_split_ratio = args.snac_split_ratio
    else:
        snac_split_ratio = 0.0

    if args.import_sf_model:
        print("Loading previous SF parameters....")
        feaNet, reward_feature_weights = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/sf_network.p", "rb")
        )
    else:
        reward_feature_weights = None
        if args.env_name in ("PointNavigation"):
            msg = colorize(
                "\nVAE Feature Extractor is selected!!!",
                "yellow",
                bold=True,
            )
            print(msg)
            feaNet = AutoEncoder(
                state_dim=args.s_dim,
                action_dim=args.a_dim,
                fc_dim=args.sf_fc_dim,
                sf_dim=args.sf_dim,
                snac_split_ratio=snac_split_ratio,
                activation=nn.ELU(),
            )
        else:
            msg = colorize(
                "\nCNN Feature Extractor is selected!!!",
                "yellow",
                bold=True,
            )
            print(msg)

            encoder_conv_layers, decoder_conv_layers = get_conv_layer(args)
            feaNet = ConvNetwork(
                state_dim=args.s_dim,
                action_dim=args.a_dim,
                encoder_conv_layers=encoder_conv_layers,
                decoder_conv_layers=decoder_conv_layers,
                fc_dim=args.sf_fc_dim,
                sf_dim=args.sf_dim,
                snac_split_ratio=snac_split_ratio,
                activation=nn.Tanh(),
            )

    algo_classes = {"SNAC": SF_SNAC, "EigenOption": SF_EigenOption, "default": SF_LASSO}

    sf_network_class = algo_classes.get(args.algo_name, SF_LASSO)

    sf_network = sf_network_class(
        env_name=args.env_name,
        feaNet=feaNet,
        feature_weights=reward_feature_weights,
        a_dim=args.a_dim,
        sf_dim=args.sf_dim,
        snac_split_ratio=snac_split_ratio,
        sf_lr=args.sf_lr,
        batch_size=args.sf_batch_size,
        reward_loss_scaler=args.reward_loss_scaler,
        state_loss_scaler=args.state_loss_scaler,
        weight_loss_scaler=args.weight_loss_scaler,
        lasso_loss_scaler=args.lasso_loss_scaler,
        is_discrete=args.is_discrete,
        sf_path=sf_path,
        device=args.device,
    )

    return sf_network


def call_opNetwork(
    sf_network: nn.Module,
    reward_options: np.ndarray,
    state_options: np.ndarray,
    args,
):
    from models.policy import OP_Controller

    if args.algo_name == "SNAC":
        snac_split_ratio = args.snac_split_ratio
    else:
        snac_split_ratio = 0.0

    if args.import_op_model:
        print("Loading previous OP parameters....")
        if args.op_mode == "sac":
            policy, critic, reward_options, state_options, alpha = pickle.load(
                open(
                    f"log/eval_log/model_for_eval/{args.env_name}/op_sac_network.p",
                    "rb",
                )
            )
        elif args.op_mode == "ppo":
            alpha = None
            policy, critic, reward_options, state_options = pickle.load(
                open(
                    f"log/eval_log/model_for_eval/{args.env_name}/op_ppo_network.p",
                    "rb",
                )
            )
    else:
        if args.op_mode == "sac":
            policy = OptionPolicy(
                input_dim=args.flat_s_dim,
                hidden_dim=args.op_policy_dim,
                a_dim=args.a_dim,
                num_weights=2 * args.num_options,
                activation=nn.ReLU(),
                is_discrete=args.is_discrete,
            )
            critic = OPtionCriticTwin(
                input_dim=args.flat_s_dim + args.a_dim,
                hidden_dim=args.op_critic_dim,
                num_weights=2 * args.num_options,
                activation=nn.ReLU(),
            )
            alpha = args.sac_init_alpha
        else:
            policy = OptionPolicy(
                input_dim=args.flat_s_dim,
                hidden_dim=args.op_policy_dim,
                a_dim=args.a_dim,
                num_weights=2 * args.num_options,
                activation=nn.Tanh(),
                is_discrete=args.is_discrete,
            )
            critic = OptionCritic(
                input_dim=args.flat_s_dim,
                hidden_dim=args.op_critic_dim,
                num_weights=2 * args.num_options,
                activation=nn.Tanh(),
            )
            alpha = None

    op_network = OP_Controller(
        sf_network=sf_network,
        policy=policy,
        critic=critic,
        sf_dim=args.sf_dim,
        snac_split_ratio=snac_split_ratio,
        reward_options=reward_options,
        state_options=state_options,
        minibatch_size=args.op_batch_size,
        alpha=alpha,
        args=args,
    )

    return op_network


def call_hcNetwork(sf_network: nn.Module, op_network: nn.Module, args):
    from models.policy import HC_Controller

    if args.import_hc_model:
        print("Loading previous HC parameters....")
        policy, primitivePolicy, critic = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/hc_network.p", "rb")
        )
    else:
        policy = HC_Policy(
            input_dim=args.flat_s_dim,
            hidden_dim=args.hc_policy_dim,
            hc_action_dim=2 * args.num_options + 1,
            activation=nn.Tanh(),
        )
        if args.PM_policy == "PPO":
            primitivePolicy = HC_PPO(
                input_dim=args.flat_s_dim,
                hidden_dim=args.hc_policy_dim,
                a_dim=args.a_dim,
                is_discrete=args.is_discrete,
                activation=nn.Tanh(),
            )
        elif args.PM_policy == "RW":
            primitivePolicy = HC_RW(
                a_dim=args.a_dim,
                is_discrete=args.is_discrete,
            )
        else:
            NotImplementedError(f"{args.PM_policy} is not implemented")

        critic = HC_Critic(
            input_dim=args.flat_s_dim,
            hidden_dim=args.hc_critic_dim,
            activation=nn.Tanh(),
        )

    policy = HC_Controller(
        sf_network=sf_network,
        op_network=op_network,
        policy=policy,
        primitivePolicy=primitivePolicy,
        critic=critic,
        minibatch_size=args.hc_batch_size,
        a_dim=args.a_dim,
        policy_lr=args.hc_policy_lr,
        critic_lr=args.hc_critic_lr,
        entropy_scaler=args.hc_entropy_scaler,
        eps=args.eps_clip,
        gae=args.gae,
        gamma=args.gamma,
        K=args.K_epochs,
        target_kl=args.target_kl,
        l2_reg=args.weight_loss_scaler,
        device=args.device,
    )

    return policy
