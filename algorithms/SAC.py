import torch
import torch.nn as nn
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import SF_Evaluator, PPO_Evaluator
from models import SFTrainer, SACTrainer
from utils import *
from utils.call_env import call_env


class SAC:
    def __init__(self, env: gym.Env, logger, writer, args):
        """
        This is a naive SAC wrapper that includes all necessary training pipelines for HRL.
        This trains the SF network and then trains SAC according to the extracted features by the SF network.
        """
        self.env = env

        # Define buffers and sampler for Monte-Carlo sampling
        self.buffer = TrajectoryBuffer(
            episode_len=args.episode_len,
            min_num_trj=args.sac_min_num_traj,
            max_num_trj=args.sac_max_num_traj,
        )
        self.sampler = OnlineSampler(
            env=self.env,
            state_dim=args.s_dim,
            feature_dim=args.sf_dim,
            action_dim=args.a_dim,
            hc_action_dim=args.num_vector + 1,
            agent_num=args.agent_num,
            min_option_length=args.min_option_length,
            num_options=1,
            min_cover_option_length=args.min_cover_option_length,
            episode_len=args.episode_len,
            batch_size=args.batch_size,
            min_batch_for_worker=args.min_batch_for_worker,
            cpu_preserve_rate=args.cpu_preserve_rate,
            num_cores=args.num_cores,
            gamma=args.gamma,
            verbose=False,
        )

        # Object initialization
        self.logger = logger
        self.writer = writer
        self.args = args

        # Parameter initialization
        self.curr_epoch = 0

        # SF checkpoint because plotter will only be used
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

        ### Define evaluators tailored for each process
        self.sac_evaluator = PPO_Evaluator(
            logger=logger,
            writer=writer,
            training_env=self.env,
            plotter=self.plotter,
            renderPlot=args.rendering,
            render_fps=args.render_fps,
            dir=self.sac_path,
            log_interval=args.sac_log_interval,
            eval_ep_num=10,
        )

    def run(self):
        self.train_sac()
        torch.cuda.empty_cache()

    def train_sac(self):
        self.sampler.initialize(
            batch_size=self.args.sac_batch_size,
            num_option=1,
            min_batch_for_worker=self.args.min_batch_for_worker,
        )
        ### Call network parameters and run
        self.sac_network = call_sacNetwork(self.args)
        print_model_summary(self.sac_network, model_name="SAC model")
        if not self.args.import_sac_model:
            sac_trainer = SACTrainer(
                policy=self.sac_network,
                sampler=self.sampler,
                buffer=self.buffer,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.sac_evaluator,
                epoch=self.curr_epoch + self.args.SAC_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.sac_step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.sac_log_interval,
                grid_type=self.args.grid_type,
            )
            final_epoch = sac_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.SAC_epoch

        self.curr_epoch += final_epoch
