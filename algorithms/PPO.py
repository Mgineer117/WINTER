import torch
import torch.nn as nn
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import SF_Evaluator, PPO_Evaluator
from models import SFTrainer, PPOTrainer
from utils import *
from utils.call_env import call_env


class PPO:
    def __init__(self, env: gym.Env, logger, writer, args):
        """
        This is a naive PPO wrapper that includes all necessary training pipelines for HRL.
        This trains SF network and train PPO according to the extracted features by SF network
        """
        self.env = env

        # define buffers and sampler for Monte-Carlo sampling
        total_batch_size = int(args.ppo_batch_size * args.K_epochs)
        self.sampler = OnlineSampler(
            env=self.env,
            state_dim=args.s_dim,
            action_dim=args.a_dim,
            hc_action_dim=2 * args.num_options + 1,
            min_option_length=args.min_option_length,
            num_options=1,
            episode_len=args.episode_len,
            batch_size=total_batch_size,
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
        self.ppo_evaluator = PPO_Evaluator(
            logger=logger,
            writer=writer,
            training_env=self.env,
            plotter=self.plotter,
            renderPlot=args.rendering,
            dir=self.ppo_path,
            eval_ep_num=args.eval_episodes,
        )

    def run(self):
        self.train_ppo()
        torch.cuda.empty_cache()

    def train_ppo(self):
        ### Call network param and run
        self.ppo_network = call_ppoNetwork(self.args)
        print_model_summary(self.ppo_network, model_name="PPO model")
        if not self.args.import_ppo_model:
            ppo_trainer = PPOTrainer(
                policy=self.ppo_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.ppo_evaluator,
                timesteps=self.curr_epoch + self.args.PPO_timesteps,
                log_interval=self.args.ppo_log_interval,
                grid_type=self.args.grid_type,
            )
            ppo_trainer.train()
        else:
            ppo_trainer.evaluate()
