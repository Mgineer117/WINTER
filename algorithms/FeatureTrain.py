import torch
import torch.nn as nn
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import (
    SF_Evaluator,
    OP_Evaluator,
    UG_Evaluator,
    HC_Evaluator,
)
from models import SFTrainer, OPTrainer, HCTrainer

from log.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

from utils import *
from utils.call_env import call_env


class SF_Train:
    def __init__(
        self,
        env: gym.Env,
        logger: WandbLogger,
        writer: SummaryWriter,
        args: ArgumentParser,
    ):
        """
        SNAC Specialized Neurons and Clustering Architecture
        ---------------------------------------------------------------------------
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
        self.curr_epoch = 0

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
        self.sf_evaluator = SF_Evaluator(
            logger=logger,
            writer=writer,
            training_env=self.env,
            plotter=self.plotter,
            dir=self.sf_path,
        )

    def train(self):
        """
        This trains the SF netowk. This includes training of CNN as feature extractor
        and Psi_rw network as a supervised approach as a evaluation metroc for later use.
        ----------------------------------------------------------------------------------
        Input:
            - None
        Return:
            - None
        """
        ### Call network param and run
        sf_network = call_sfNetwork(self.args, self.sf_path)
        lr_step_size = self.args.SF_epoch // 10
        scheduler = torch.optim.lr_scheduler.StepLR(
            sf_network.feature_optims, step_size=lr_step_size, gamma=0.9
        )
        print_model_summary(sf_network, model_name="SF model")
        if not self.args.import_sf_model:
            sf_trainer = SFTrainer(
                policy=sf_network,
                sampler=self.sampler,
                buffer=self.buffer,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.sf_evaluator,
                scheduler=scheduler,
                epoch=self.args.SF_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.sf_log_interval,
                post_process=self.args.post_process,
                grid_type=self.args.grid_type,
            )
            final_epoch = sf_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.SF_epoch

        self.curr_epoch += final_epoch

        return sf_network, self.curr_epoch
