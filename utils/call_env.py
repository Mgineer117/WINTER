import safety_gymnasium as sgym
import gymnasium as gym

from safety_gymnasium import __register_helper
from gym_multigrid.envs.oneroom import OneRoom
from gym_multigrid.envs.maze import Maze
from gym_multigrid.envs.lavarooms import LavaRooms
from gym_multigrid.envs.ctf import CtF


from utils.wrappers import GridWrapper, CtFWrapper, NavigationWrapper, GymWrapper


def disc_or_cont(env, args):
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.is_discrete = True
        print(f"Discrete Action Space")
    elif isinstance(env.action_space, gym.spaces.Box):
        args.is_discrete = False
        print(f"Continuous Action Space")
    else:
        raise ValueError(f"Unknown action space type {env.action_space}.")


def call_env(args):
    # define the env
    if args.env_name == "OneRoom":
        # first call dummy env to find possible location for agent
        env = OneRoom(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return GridWrapper(env, args)
    elif args.env_name == "LavaRooms":
        # first call dummy env to find possible location for agent
        env = LavaRooms(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return GridWrapper(env, args)
    elif args.env_name in ("CtF"):
        # if args.ctf_map == "sparse":
        #     if args.env_name == "CtF1v1":
        #         map_path: str = "assets/sparse_ctf_1v1.txt"
        #     elif args.env_name == "CtF1v2":
        #         map_path: str = "assets/sparse_ctf_1v2.txt"
        # elif args.ctf_map == "regular":
        map_path: str = "assets/regular_ctf.txt"
        # else:
        #     raise NotImplementedError(f"{map_path} not implemented")
        observation_option: str = "tensor"
        env = CtF(
            map_path=map_path,
            observation_option=observation_option,
            territory_adv_rate=1.0,
            max_steps=args.episode_len,
            battle_reward_ratio=0.15,
            step_penalty_ratio=0.0,
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return CtFWrapper(env, args)
    elif args.env_name == "PointNavigation":
        config = {"agent_name": "Point"}
        env_id = "PointNavigation"
        __register_helper(
            env_id=env_id,
            entry_point="gym_continuous.env_builder:Builder",
            spec_kwargs={"config": config, "task_id": env_id},
            max_episode_steps=args.episode_len,
        )

        env = sgym.make(
            "PointNavigation",
            render_mode="rgb_array",
            width=1024,
            height=1024,
            max_episode_steps=args.episode_len,
            camera_name="fixedfar",
        )

        disc_or_cont(env, args)
        args.agent_num = 1
        return NavigationWrapper(env, args)
    elif args.env_name == "InvertedPendulum":
        env = gym.make(
            "InvertedPendulum-v4",
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = 1
        return GymWrapper(env, args)
    elif args.env_name == "Hopper":
        env = gym.make(
            "Hopper-v4",
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = 1
        return GymWrapper(env, args)
    else:
        raise ValueError(f"Invalid environment key: {args.env_name}")
