# Unhinged main.py
# import networkx
# import networkx as nx
# import shapely as sp
# from shapely.geometry import Polygon
# import Geometry3D

import numpy as np
from scipy.spatial import ConvexHull
from enum import Enum

class Axis(Enum):
    X = [1,0,0]
    Y = [0,1,0]
    Z = [0,0,1]

"""
Static helper functions for linear algebra
"""


def normalize(vec1):
    len = np.linalg.norm(vec1)
    return vec1 / len


def is_rectangle(coords):
    # Check if coordinates actually create a rectangle shape
    hvec = coords[1] - coords[0]
    wvec = coords[3] - coords[0]

    dotprod = np.dot(normalize(hvec), normalize(wvec))

    if (abs(dotprod) > 0.002):
        # if the dot product between h and w is not 0, then they are not perpendicular lines of a rectangle.
        print("Input shape interior dot product: ")
        print(dotprod)
        return False
    else:
        return True


"""
Patch: Our proxy for a rectangle and only contains a rectangle
"""


class Patch:
    id_incr = 0
    def __init__(self, rect_coords):
        if not (is_rectangle(rect_coords)):
            raise Exception("Input is not a rectangle! Might be a trapezoid or parallelogram")

        print("Input is rectangle...")
        # 3d coordinates corresponding to a rectangle
        self.coords = rect_coords
        # a point on the plane, used for SDF
        self.constant = np.mean(rect_coords, axis=0)
        # calculate the normal of the surface immediately
        self.normal = self.calc_normal()
        # calculate area
        self.area = self.calc_area()

        # id for debug purposes
        self.id = self.id_incr
        self.id_incr += 1


    def calc_area(self):
        # calculates the area of the patch
        h = np.linalg.norm(self.coords[1] - self.coords[0])
        w = np.linalg.norm(self.coords[3] - self.coords[0])

        # add check for whether the points are coplanar
        return h * w

    def calc_normal(self):
        # calculates the normal of the patch
        # it can have two norms: and they're the negative of each other, but this function only returns one
        hvec = self.coords[1] - self.coords[0]
        wvec = self.coords[3] - self.coords[0]

        # Take the cross product, then normalize
        surf = np.cross(hvec, wvec)
        return normalize(surf)

    def calc_intersection(self, other):
        print("calc_intersection: implement me")

    def signed_distance_to(self, vec):
        dv = vec - self.constant
        return np.dot(dv, self.normal)

    def get_corner_pts(self):
        return self.coords


"""
Modification: A combination of a shrinkage and a number of hinges
"""


class Modification:
    def __init__(self, num_hinges, scale, cost):
        self.cost = cost

        # Shrinking variables
        self.scale = scale
        # self.position = pos

        # Hinge variables
        self.numSplits = num_hinges

        # region
        self.projected_region = self.calc_region()

    def calc_region(self):
        print("calc_region: implement me")
        return -1


"""
FoldConfig: A list of i angles corresponding to i hinges
Notes: May not actually need this as a class
"""


class FoldConfig:
    def __init__(self):
        # A list of i angles corresponding to i hinges
        self.angles = []  # think of it as an empty numpy array
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
    def __init__(self, isleft, mod, patch_list):
        self.modification = mod
        self.isleft = isleft
        self.fold_transform = None
        self.rot_axis = []

        # this patch list should be at least size 2
        self.patch_list = patch_list

    def gen_fold_transform(self):
        print("gen_fold_transform: implement me!")
        # Based on the modification, generate start angle and end angle
        # TODO: Di or David

        #crossing the normals of base patch and normal patch
        self.rot_axis = np.cross(self.patch_list[1].calc_normal(), self.patch_list[0].calc_normal())

        #using patch_list to determine t or h scaffold
        num_hinges = self.modification.numSplits + len(self.patch_list) - 1

        start_config = FoldConfig()
        end_config = FoldConfig()

        for i in range(0, num_hinges):
            
            start_ang = 0.0
            end_ang = 0.0
            #first base patch and foldable patch connection
            if i == 0:
                start_ang = 0.0
                end_ang = 90.0

            #final hinge, different based on if t or h
            elif i == num_hinges - 1 and len(self.patch_list) == 3:
                start_ang = 0.0
                end_ang = -90.0
            #even hinges
            elif i % 2 == 0:
                start_ang = 0.0
                end_ang = 180.0
            #odd hinges
            else:
                start_ang = 0.0
                end_ang = -180.0
            
            if not self.isleft:
                start_config.angles.append(start_ang * -1)
                end_config.angles.append(end_ang * -1)
            else:
                start_config.angles.append(start_ang)
                end_config.angles.append(end_ang)

        self.fold_transform = FoldTransform(start_config, end_config)
"""
BasicScaff: Parent class for TBasicScaff and HBasicScaff
"""


class BasicScaff():
    def __init__(self):
        self.fold_options = []

"""
TBasicScaff: A basic scaffold of type T
"""


class TBasicScaff(BasicScaff):
    def __init__(self, f_patch, b_patch):
        super().__init__()
        self.f_patch = f_patch
        self.b_patch = b_patch

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        patch_list = [self.f_patch, self.b_patch]
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                if (i % 2 != 0) and (i <= j):
                    print("start")
                    # if odd number of hinges
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j, cost)
                    fo_left = FoldOption(True, mod, patch_list)
                    fo_right = FoldOption(False, mod, patch_list)

                    # generate the fold transform from start to end?
                    fo_left.gen_fold_transform()
                    fo_right.gen_fold_transform()

                    self.fold_options.append(fo_left)
                    self.fold_options.append(fo_right)

"""
HBasicScaff: A basic scaffold of type H
"""


class HBasicScaff(BasicScaff):
    def __init__(self, f_patch, b_patch_low, b_patch_high):
        super().__init__()
        self.f_patch = f_patch
        self.b_patch_low = b_patch_low
        self.b_patch_high = b_patch_high

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        patch_list = [self.f_patch, self.b_patch_low, self.b_patch_high]
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                if (i % 2 != 0) and (i <= j):
                    # if odd number of hinges
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j, cost)
                    fo_left = FoldOption(True, mod, patch_list)
                    fo_right = FoldOption(False, mod, patch_list)

                    # generate the fold transform from start to end?
                    fo_left.gen_fold_transform()
                    fo_right.gen_fold_transform()

                    self.fold_options.append(fo_left)
                    self.fold_options.append(fo_right)


"""
MidScaff: a mid level folding unit that contains basic scaffolds
"""


class MidScaff:
    def __init__(self, bs):
        self.basic_scaffs = bs
        self.conflict_graph = None

    def gen_conflict_graph(self):
        print("gen_conflict_graph: implement me")


"""
InputScaff: The full input scaff
"""


class InputScaff:
    def __init__(self, p):
        self.patches = p
        self.hinge_graph = None
        self.mid_scaffs = []

        # debug purposes for ease of our test algorithm
        # for now we manually define basic scaffolds
        self.basic_scaffs = []

    def gen_scaffs(self):
        # TODO: Di
        # generates basic scaffolds
        # generates hinge graph
        # generates mid-level scaffolds
        print("gen_scaffs: implement me")
        

    def gen_hinge_graph(self):
        # TODO: Di
        print("gen_hinge_graph...")
        self.hinge_graph = networkx.Graph()
        for bs in self.basic_scaffs:
            self.hinge_graph.add_node(bs.f_patch.id)
            if (bs.b_patch_low): # if this exists, then H scaffold
                self.hinge_graph.add_node(bs.b_patch_low.id)
                self.hinge_graph.add_edge(bs.f_patch.id, bs.b_patch_low.id)

                self.hinge_graph.add_node(bs.b_patch_high.id)
                self.hinge_graph.add_edge(bs.f_patch.id, bs.b_patch_high.id)
            else: # otherwise, T scaffold
                self.hinge_graph.add_node(bs.b_patch.id)
                self.hinge_graph.add_edge(bs.f_patch.id, bs.b_patch.id)


def patch_test():
    # TODO: Di
    # Create the input Scaffold
    # Generate patches from the input mesh, for testing purposes, just generate some rectangles
    # They have to be around the polygon in the right order.

    print("creating rectangle 1...")
    coords = np.array([(0, 0, 0), (0, 2, 0), (4, 2, 2), (4, 0, 2)])
    rect = Patch(coords)
    print(rect.calc_area())  # expected 2 * root(20) = 8.944
    print(rect.calc_normal())  # expected 4/root(70), 0, -8/root(70)

    print("creating rectangle 3...")
    coords2 = np.array([(0, 0, 2), (0, 4, 2), (2, 4, 0), (2, 0, 0)])
    rect2 = Patch(coords2)
    print(rect2.calc_area())
    print(rect2.calc_normal())


def basic_t_scaffold():
    # coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)]) # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)]) # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)]) # foldable patch

    foldable = Patch(coords3)
    base = Patch(coords2)

    tscaff = TBasicScaff(foldable, base)
    tscaff.gen_fold_options(1, 1, .5)
    print("Begin test")

    for scaff in tscaff.fold_options:
        for start in scaff.fold_transform.startAngles.angles:
            print(start)
        for end in scaff.fold_transform.endAngles.angles:
            print(end)
        print('------------------------------')

def basic_h_scaffold():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)]) # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)]) # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)]) # foldable patch

    foldable = Patch(coords3)
    base = Patch(coords2)
    top = Patch(coords1)

    tscaff = HBasicScaff(foldable, base, top)
    tscaff.gen_fold_options(1, 1, .5)
    print("Begin test")

    for scaff in tscaff.fold_options:
        for start in scaff.fold_transform.startAngles.angles:
            print(start)
        for end in scaff.fold_transform.endAngles.angles:
            print(end)
        print('------------------------------')

basic_h_scaffold()