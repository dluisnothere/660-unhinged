# Unhinged main.py
import networkx as nx
import shapely as sp
from shapely.geometry import Polygon
import numpy as np
import Geometry3D
from scipy.spatial import ConvexHull


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
    def __init__(self, rect_coords):
        if not (is_rectangle(rect_coords)):
            raise Exception("Input is not a rectangle! Might be a trapezoid or parallelogram")

        # 3d coordinates corresponding to a rectangle
        self.coords = rect_coords
        # a point on the plane, used for SDF
        self.constant = np.mean(rect_coords, axis=0)
        # calculate the normal of the surface immediately
        self.normal = self.calc_normal()
        # calculate area
        self.area = self.calc_area()

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
        self.mid_scaffs = None

    def gen_scaffs(self):
        # TODO: Di
        # generates basic scaffolds
        # generates mid-level scaffolds
        print("gen_scaffs: implement me")

    def gen_hinge_graph(self):
        # TODO: Di
        print("gen_hinge_graph: implement me")


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

    # print("creating rectangle 2... should throw an error")
    # coords2 = np.array([(3, 0, 2), (0, 4, 2), (1, 4, 0), (4, 0, 0)])
    # rect2 = Patch(coords2)
    # print(rect2.calc_area())
    # print(rect2.calc_normal())


def intersection_test():

    # print("creating rectangle 1...")
    coords = np.array([(0, 0, 0), (0, 4, 0), (4, 4, 0), (4, 0, 0)])
    rect = Patch(coords)
    print(rect.calc_area())  # expected 2 * root(20) = 8.944
    print(rect.calc_normal())  # expected 4/root(70), 0, -8/root(70)

    print("creating rectangle 2...")
    coords2 = np.array([(0, 2, 0), (4, 2, 0), (4, 2, 4), (0, 2, 4)])
    rect2 = Patch(coords2)
    print(rect2.calc_area())
    print(rect2.calc_normal())

    rect.calc_intersection(rect2)

def has_intersection(foldable, base, thr):

    def get_distance(foldable, base):
        # unexpectedly complicated and difficult for two finite planes
        print("get_distance: figure me out")

    if get_distance(foldable, base) > thr:
        return False

    left_end = np.inf
    right_end = -np.inf

    for p in foldable.get_corner_pts():
        sd = base.signed_distance_to(p)
        left_end = min(left_end, sd)
        right_end = max(right_end, sd)

    if left_end * right_end >= 0:
        return False
    else:
        shorter_end = min(abs(left_end), abs(right_end))
        return shorter_end > thr