import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from algorithms import SF_Train
from log.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from models.evaulators import (
    OP_Evaluator,
    HC_Evaluator,
)

from models import OPTrainer, HCTrainer
from utils import *
from utils.call_weights import get_reward_maps, call_options


class EigenOption:
    def __init__(
        self,
        env: gym.Env,
        logger: WandbLogger,
        writer: SummaryWriter,
        args,
    ):
        """
        SNAC Specialized Neurons and Clustering Architecture
        -----s----------------------------------------------------------------------
        There are several improvements
            - Clustering in Eigenspace
            - Simultaneous Reward and State Decomposition

        Training Pipeline:
            - Train SF network: train CNN (feature extractor)
            - Obtain Reward and State eigenvectors
            - Train OP network (Option Policy) for each of intrinsic reward by eigenvectors
                - PPO is used to train OP network
            - Train HC network (Hierarchical Controller that alternates between option and random walk)
        """
        self.env = env

        # define buffers and sampler for Monte-Carlo sampling
        self.buffer = TrajectoryBuffer(
            state_dim=args.s_dim,
            action_dim=args.a_dim,
            hc_action_dim=2 * args.num_options + 1,
            episode_len=args.episode_len,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
        )
        self.sampler = OnlineSampler(
            env=self.env,
            state_dim=args.s_dim,
            action_dim=args.a_dim,
            hc_action_dim=2 * args.num_options + 1,
            min_option_length=args.min_option_length,
            num_options=1,
            episode_len=args.episode_len,
            batch_size=args.warm_batch_size,
            min_batch_for_worker=args.min_batch_for_worker,
            cpu_preserve_rate=args.cpu_preserve_rate,
            num_cores=args.num_cores,
            gamma=args.gamma,
            verbose=False,
        )

        # object initialization
        self.logger = logger
        self.writer = writer
        self.args = args

        # param initialization
        self.curr_timesteps = args.SF_epoch * args.step_per_epoch

        # SF checkpoint b/c plotter will only be used
        (
            self.sf_path,
            self.op_path,
            self.hc_path,
            self.oc_path,
            self.ppo_path,
            self.sac_path,
        ) = self.logger.checkpoint_dirs

        self.plotter = Plotter(
            grid_size=args.grid_size,
            img_tile_size=args.img_tile_size,
            sf_path=self.sf_path,
            op_path=self.op_path,
            hc_path=self.hc_path,
            oc_path=self.oc_path,
            sac_path=self.sac_path,
            ppo_path=self.ppo_path,
            log_dir=logger.log_dir,
            device=args.device,
        )

        ### Define evaulators tailored for each process
        # each evaluator has slight deviations
        evaluator_params = {
            "logger": logger,
            "writer": writer,
            "training_env": self.env,
            "plotter": self.plotter,
            "gridPlot": True,
            "renderPlot": args.rendering,
            "eval_ep_num": args.eval_episodes,
        }

        if args.env_name in ("PointNavigation"):
            evaluator_params.update({"gridPlot": False})

        self.op_evaluator = OP_Evaluator(dir=self.op_path, **evaluator_params)
        self.hc_evaluator = HC_Evaluator(
            dir=self.hc_path,
            min_option_length=args.min_option_length,
            **evaluator_params,
        )

    def run(self):
        self.train_sf()
        torch.cuda.empty_cache()
        self.train_op()
        torch.cuda.empty_cache()
        self.train_hc()
        torch.cuda.empty_cache()

    def train_sf(self):
        ft = SF_Train(
            env=self.env,
            logger=self.logger,
            writer=self.writer,
            args=self.args,
        )
        self.sf_network, _ = ft.train()

        reward_options, state_options = call_options(
            algo_name=self.args.algo_name,
            sf_dim=self.args.sf_dim,
            snac_split_ratio=self.args.snac_split_ratio,
            temporal_balance_ratio=self.args.temporal_balance_ratio,
            num_options=self.args.num_options,
            sf_network=self.sf_network,
            sampler=ft.sampler,
            buffer=ft.buffer,
            DIF_batch_size=self.args.DIF_batch_size,
            grid_type=self.args.grid_type,
            gamma=self.args.gamma,
            method=self.args.method,
            device=self.args.device,
        )

        self.reward_options = reward_options
        self.state_options = state_options

        if self.args.env_name in ("OneRoom", "FourRooms", "Maze", "CtF"):
            images = get_reward_maps(
                env=self.env,
                sf_network=self.sf_network,
                V=[reward_options, state_options],
                feature_dim=self.args.sf_dim,
                grid_type=self.args.grid_type,
            )
            self.logger.write_images(
                step=self.curr_timesteps, images=images, log_dir="RewardMap/Options"
            )

    def train_op(self):
        """
        This discovers the eigenvectors via clustering for each of reward and state decompositions.
        --------------------------------------------------------------------------------------------
        """
        total_batch_size = int(self.args.op_batch_size * self.args.K_epochs)
        self.sampler.initialize(
            batch_size=total_batch_size,
            num_option=2 * self.args.num_options,
            min_batch_for_worker=self.args.op_min_batch_for_worker,
        )

        if not self.args.import_op_model:
            self.op_network = call_opNetwork(
                self.sf_network, self.reward_options, self.state_options, self.args
            )
            print_model_summary(self.op_network, model_name="OP model")

            op_trainer = OPTrainer(
                policy=self.op_network,
                sampler=self.sampler,
                buffer=self.buffer,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.op_evaluator,
                num_weights=2 * self.args.num_options,
                mode=self.args.op_mode,
                timesteps=self.args.OP_timesteps,
                init_timesteps=self.curr_timesteps,
                batch_size=self.args.op_batch_size,
                log_interval=self.args.op_log_interval,
                grid_type=self.args.grid_type,
            )

            self.curr_timesteps = op_trainer.train()
        else:
            self.op_network = call_opNetwork(
                self.sf_network, self.reward_options, self.state_options, self.args
            )
            self.curr_timesteps = self.curr_timesteps + self.args.OP_timesteps

    def train_hc(self):
        """
        Train Hierarchical Controller to compute optimal policy that alternates between
        options and the random walk.
        """
        total_batch_size = int(self.args.hc_batch_size * self.args.K_epochs / 2)
        self.sampler.initialize(
            batch_size=total_batch_size,
            num_option=1,
            min_batch_for_worker=self.args.min_batch_for_worker,
        )

        self.hc_network = call_hcNetwork(self.sf_network, self.op_network, self.args)
        print_model_summary(self.hc_network, model_name="HC model")
        if not self.args.import_hc_model:
            hc_trainer = HCTrainer(
                policy=self.hc_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.hc_evaluator,
                timesteps=self.args.HC_timesteps,
                init_timesteps=self.curr_timesteps,
                log_interval=self.args.hc_log_interval,
                grid_type=self.args.grid_type,
            )
            hc_trainer.train()
