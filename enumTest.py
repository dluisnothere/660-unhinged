from __future__ import annotations
import networkx as nx
import numpy as np
from enum import Enum
from typing import Dict, List, Set

class Axis(Enum):
    X = np.array([-1, 0, 0])
    Y = np.array([0, -1, 0])
    Z = np.array([0, 0, -1])
