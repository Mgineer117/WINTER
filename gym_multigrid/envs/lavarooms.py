from itertools import chain
from typing import Final, Literal, TypeAlias, TypedDict
from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import random
from numpy.typing import NDArray

from gym_multigrid.core.constants import *
from gym_multigrid.utils.window import Window
from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, MazeActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall, Lava, Key
from gym_multigrid.core.world import RoomWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position


class ObservationDict(TypedDict):
    blue_agent: NDArray[np.int_]
    red_agent: NDArray[np.int_]
    blue_flag: NDArray[np.int_]
    red_flag: NDArray[np.int_]
    blue_territory: NDArray[np.int_]
    red_territory: NDArray[np.int_]
    obstacle: NDArray[np.int_]
    is_red_agent_defeated: int


class MultiAgentObservationDict(TypedDict):
    blue_agent: NDArray[np.int_]
    red_agent: NDArray[np.int_]
    blue_flag: NDArray[np.int_]
    red_flag: NDArray[np.int_]
    blue_territory: NDArray[np.int_]
    red_territory: NDArray[np.int_]
    obstacle: NDArray[np.int_]
    terminated_agents: NDArray[np.int_]


Observation: TypeAlias = ObservationDict | MultiAgentObservationDict | NDArray[np.int_]


class LavaRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int = 0,
        width=12,
        height=12,
        max_steps: int = 100,
        agent_view_size: int = 7,
        tile_size: int = 20,
        highlight_visible_cells: bool | None = True,
        partial_observability: bool = False,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        self.grid_type = grid_type

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.world = RoomWorld
        self.actions_set = MazeActions

        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="agent",
            )
        ]

        self.goal_positions = [[(8, 8)]]  # (7, 1)
        self.agent_positions = [(3, 3)]
        self.lava_positions = [[(8, 5)]]
        self.key_positions = [[(5, 8)]]

        self.map_structure = [
            "############",
            "#    #     #",
            "#          #",
            "#          #",
            "#    #     #",
            "##  ##     #",
            "#          #",
            "#          #",
            "#          #",
            "#          #",
            "#          #",
            "############",
        ]

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            partial_obs=partial_observability,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def _gen_grid(self, width, height, options):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.map_structure):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)

        # goal allocation
        for i, pos in enumerate(self.goal_positions[self.grid_type]):
            goal = Goal(self.world, i)
            self.put_obj(goal, *pos)

        # lava allocation
        for pos in self.lava_positions[self.grid_type]:
            lava = Lava(self.world)
            self.put_obj(lava, *pos)

        # key allocation
        for pos in self.key_positions[self.grid_type]:
            key = Key(self.world)
            self.put_obj(key, *pos)

        # agent allocation
        if options["random_init_pos"]:
            coords = self.find_obj_coordinates(None)
            agent_positions = random.sample(coords, 1)[0]
        else:
            agent_positions = self.agent_positions[self.grid_type]

        for agent in self.agents:
            self.place_agent(agent, pos=agent_positions)

    def find_obj_coordinates(self, obj) -> tuple[int, int] | None:
        """
        Finds the coordinates (i, j) of the first occurrence of None in the grid.
        Returns None if no None value is found.
        """
        coord_list = []
        for index, value in enumerate(self.grid.grid):
            if value is obj:
                # Calculate the (i, j) coordinates from the 1D index
                i = index % self.width
                j = index // self.width
                coord_list.append((i, j))
        return coord_list

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = {"image": obs[0][:, :, 0:1]}
        return observations, info

    def get_rewards(self, fwd_cell, rewards):
        if fwd_cell.type == "goal":
            done = True
            rewards += 1.0 - 0.9 * (self.step_count / self.max_steps)
            # rewards += -0.5
            # rewards += 0.0
        elif fwd_cell.type == "lava":
            done = True
            #  rewards += 1.0 - 0.9 * (self.step_count / self.max_steps)
            rewards += -0.5
            # rewards += 0.0
        elif fwd_cell.type == "key":
            done = True
            # rewards += 1.0 - 0.9 * (self.step_count / self.max_steps)
            # rewards += -0.5
            rewards += 0.0
        else:
            done = False

        return rewards, done

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))

        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the current agent position
            curr_pos = self.agents[i].pos
            done = False

            # Rotate left
            if actions[i] == self.actions.left:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    rewards, done = self.get_rewards(fwd_cell, rewards)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards[i] += 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    rewards, done = self.get_rewards(fwd_cell, rewards)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards[i] += 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    rewards, done = self.get_rewards(fwd_cell, rewards)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards[i] += 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    rewards, done = self.get_rewards(fwd_cell, rewards)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                else:
                    rewards[i] += 0
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(actions))
            ]

        obs = [self.world.normalize_obs * ob for ob in obs]

        ### NOTE: not multiagent
        observations = {"image": obs[0][:, :, 0:1]}

        return observations, rewards, terminated, truncated, {}
