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
"""Wall."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


@dataclass
class Walls(Geom):  # pylint: disable=too-many-instance-attributes
    """Walls - barriers in the environment not associated with any constraint.

    # NOTE: this is probably best to be auto-generated than manually specified.
    """

    name: str = 'walls'
    num: int = 4  # Number of walls
    locate_factor: float = 1.2
    placements: list = None  # This should not be used
    locations: list = field(default_factory=list)  # This should be used and length == walls_num
    keepout: float = 0.0  # This should not be used

    color: np.array = COLOR['wall']
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = False
    is_constrained: bool = False

    def __post_init__(self):
        # Additional initialization logic for Walls
        assert self.num in (2, 4), 'Walls are specific for Circle and Run tasks.'
        assert self.locate_factor >= 0, 'Locate factor must be >= 0.'
        self.locations = [
            (self.locate_factor, 0),
            (-self.locate_factor, 0),
            (0, self.locate_factor),
            (0, -self.locate_factor),
        ]
        self.index: int = 0
    
    def get_config(self, xy_pos, rot):
        """Override to implement custom logic for generating the config."""
        geom = {
            'name': self.name,
            'size': np.array([0.05, 3.5, 0.3]),  # Custom size
            'pos': np.r_[xy_pos, 0.25],  # Position with custom z-coordinate
            'rot': 0,  # Default rotation
            'type': 'box',  # Type of the geometry (can be 'box', 'sphere', etc.)
            'group': self.group,  # Grouping information
            'rgba': self.color * [1, 1, 1, 0.1],  # Transparency and color
        }
        
        if self.index >= 2:  # Add some rotation based on index
            geom.update({'rot': np.pi / 2})
        
        self.index_tick()  # Update the index
        return geom
    
    def index_tick(self):
        """Increments and wraps the index."""
        self.index += 1
        self.index %= self.num

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""