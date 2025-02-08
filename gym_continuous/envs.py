# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Button task 1."""
# from safety_gymnasium.assets.geoms.walls import Walls
from gym_continuous.bases.base_task import BaseTask
from gym_continuous.assets.geoms import Walls
from safety_gymnasium.assets.geoms import Goal, Pillars
from safety_gymnasium.assets.mocaps import Gremlins


class PointNavigationEnv(BaseTask):
    """An agent must press a goal button while avoiding hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.0, -1.0, 1.0, 1.0]

        self.agent.locations = [(-0.85, -0.85)]
        self.agent.keepout = 0.1

        self._add_geoms(Goal(keepout=0.1, locations=[(-0.85, 0.85)]))
        self._add_geoms(Walls(num=4, locate_factor=1.25))
        self._add_geoms(
            Pillars(
                num=8,
                size=0.1,
                height=0.1,
                keepout=0.1,
                locations=[
                    (-1.0, -0.5),
                    (-0.8, -0.5),
                    (-0.6, -0.5),
                    (-0.4, -0.5),
                    (-0.2, -0.5),
                    (0.6, -0.5),
                    (0.8, -0.5),
                    (1.0, -0.5),
                ],
                is_constrained=False,
            ),
        )
        self._add_mocaps(
            Gremlins(
                num=1,
                placements=[(-0.4, -0.2, 0.2, 0.5)],
                travel=0.4,
                keepout=0.3,
                contact_cost=0.05,
            )
        )

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        # for sparse reward settings, there is no distance-wise rewards
        # reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size
