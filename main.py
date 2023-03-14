# Unhinged main.py
import networkx as nx
import shapely as sp
from shapely.geometry import Polygon
import numpy as np

"""
Patch: Our proxy for a rectangle and only contains a rectangle
"""


class Patch:
    def __init__(self, rect):
        self.rectangle = rect

"""
Modification: A combination of a shrinkage and a number of hinges
"""

class Modification:
    def __init__(self, region):
        self.cost = 0
        self.projected_region = region

        # Shrinking variables
        self.scale = 1
        self.position = np.array([0, 0, 0])

        # Hinge variables
        self.numSplits = 0


"""
FoldConfig: A list of i angles corresponding to i hinges
Notes: May not actually need this as a class
"""


class FoldConfig:
    def __init__(self):
        # A list of i angles corresponding to i hinges
        self.angles = None  # think of it as an empty numpy array
        self.time = 0


"""
FoldTransform: An association of start and end angles with their associated time step
"""


class FoldTransform:
    def __init__(self, a_st, a_en):
        # FoldConfig
        self.startAngles = a_st
        # FoldConfig
        self.endAngles = a_en


"""
FoldOption: An object that contains a modification and the associated fold transform
"""


class FoldOption:
    def __init__(self):
        self.modification = None
        self.foldTransform = None


"""
BasicScaff: Parent class for TBasicScaff and HBasicScaff
"""


class BasicScaff():
    def __init__(self, fold_options):
        self.fold_options = fold_options


"""
TBasicScaff: A basic scaffold of type T
"""


class TBasicScaff(BasicScaff):
    def __init__(self, f_patch, b_patch):
        super.__init__(None)
        self.f_patch = f_patch
        self.b_patch = b_patch

    def gen_fold_options(self, ns, nh):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("genFoldOptions: implement me")
        options = []
        # TODO: David


    def gen_cuts(self, ns):
        print("gen_cuts: implement me")

    def gen_hinges(self, nh):
        print("gen_hinges: implement me")

"""
MidScaff: a mid level folding unit that contains basic scaffolds
"""

class MidScaff:
    def __init__(self):
        self.basic_scaffs = None
        self.conflict_graph = None

    def gen_conflict_graph(self):
        print("gen_conflict_graph: implement me")

"""
InputScaff: The full input scaff
"""

class InputScaff:
    def __init__(self):
        self.hinge_graph = None
        self.mid_scaffs = None

    def gen_scaffs(self):
        # TODO: Di

    def gen_hinge_graph(self):
        # TODO: Di
        print("gen_hinge_graph: implement me")

def test():
    # TODO: Di