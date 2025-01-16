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
from gym_multigrid.core.object import Goal
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


class FourRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int = 0,
        grid_size: tuple = (13, 13),
        agent_view_size: int = 7,
        max_steps: int = 100,
        highlight_visible_cells: bool | None = True,
        tile_size: int = 20,
        partial_observability: bool = False,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------

        """
        if grid_type < 0 or grid_type >= 2:
            raise ValueError(
                f"The Fourroom only accepts grid_type of 0 and 1, given {grid_type}"
            )
        else:
            self.grid_type = grid_type

        self.width = grid_size[0]
        self.height = grid_size[1]
        self.grid_size = grid_size
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

        self.doorway_positions = [(3, 6), (6, 2), (10, 6), (7, 9)]
        self.vert_wall_positions = [(6, 0), (7, 6)]
        self.hor_wall_positions = [(0, 6), (6, 6)]

        self.goal_positions = [
            (3, 9),
            (7, 9),
        ]  # (7, 1)
        self.agent_positions = [
            (9, 3),
            (11, 1),
        ]

        self.grids = {}
        self.grid_imgs = {}

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

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # Bottom wall and door
        for coord in self.vert_wall_positions:
            self.grid.vert_wall(coord[0], coord[1], room_h)

        # Bottom wall and door
        for coord in self.hor_wall_positions:
            self.grid.horz_wall(coord[0], coord[1], room_w)

        for pos in self.doorway_positions:
            self.grid.set(*pos, None)

        # place goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.grid_type]

        # place agent
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

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"success": False}
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
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self.agents[i].pos = fwd_pos
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

        return observations, rewards, terminated, truncated, info
