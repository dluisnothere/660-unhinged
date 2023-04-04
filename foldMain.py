# Unhinged main.py
import networkx
import networkx as nx
import numpy as np
from enum import Enum

class Axis(Enum):
    X = [1,0,0]
    Y = [0,1,0]
    Z = [0,0,1]

class PatchType(Enum):
    Base = 0
    Fold = 1

"""
Static helper functions for linear algebra
"""


def normalize(vec1):
    len = np.linalg.norm(vec1)
    return vec1 / len

# Might not even need this for now
def is_rectangle(coords):
    print("Checking if input is a rectangle...")
    print("coords1:")
    print(coords[1])
    print("coords0:")
    print(coords[0])
    print("coords3:")
    print(coords[3])

    # Check if coordinates actually create a rectangle shape
    hvec = np.array(coords[1]) - np.array(coords[0])
    wvec = np.array(coords[3]) - np.array(coords[0])

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
        # if not (is_rectangle(rect_coords)):
        #     raise Exception("Input is not a rectangle! Might be a trapezoid or parallelogram")
        # print("Input is rectangle...")
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

        # patch type:
        self.patch_type = None


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
    def __init__(self, num_hinges, range_start, range_end, num_pieces, cost):
        self.cost = cost

        # Shrinking variables
        # self.scale = scale
        # num_splits is a substitute for scale, but for me it makes more sense for now.

        #start of range
        self.range_start = range_start
        #end of range
        self.range_end = range_end
        #number of pieces
        self.num_pieces= num_pieces

        # Hinge variables
        self.num_hinges = num_hinges

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
        print("Entered gen_fold_transform...")
        # Based on the modification, generate start angle and end angle
        # TODO: Di or David

        #crossing the normals of base patch and normal patch
        self.rot_axis = np.cross(self.patch_list[1].calc_normal(), self.patch_list[0].calc_normal())

        #using patch_list to determine t or h scaffold
        # num_hinges = self.modification.numSplits + len(self.patch_list) - 1
        num_hinges = self.modification.num_hinges

        start_config = [] # FoldConfig()
        end_config = [] # FoldConfig()

        # When num_hinges is 0, it means we still have the hinge at the base patch(es)
        # meaning we always have 1 more than we expect
        for i in range(0, num_hinges + 1):
            
            start_ang = 0.0
            end_and = 0.0
            #first base patch and foldable patch connection
            # Will always be a 90 degree rotation.
            if i is 0:
                start_ang = 0.0
                end_ang = 90.0

            #final hinge, different based on if t or h
            # In the test case it should not even be hitting these
            #even hinges
            elif i % 2 is 0:
                start_ang = 0.0
                end_ang = 180.0
            #odd hinges
            else:
                start_ang = 0.0
                end_ang = -180.0
            
            if not self.isleft:
                start_ang *= -1
                end_and *= -1

            # Add the angles to the start and end configs
            start_config.append(start_ang)
            end_config.append(end_ang)

            # start_config.angles.append(start_ang)
            # end_config.angles.append(end_ang)

        # If isHScaff is True, then we need to add the final hinge
        # if isHScaff:
        #     start_config.append(0.0)
        #     end_config.append(-90.0)

        self.fold_transform = FoldTransform(start_config, end_config)
"""
BasicScaff: Parent class for TBasicScaff and HBasicScaff
"""


class BasicScaff():
    id_incr = 0
    
    def __init__(self):
        self.fold_options = []
        self.id = self.id_incr
        self.id_incr += 1

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
                for k in range(0, j):
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j / ns, k / ns, j-k, cost)
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
                for k in range(0, j):
                    print(k)
                    print(j)
                    print("end")
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j / ns, k / ns, j-k, cost)
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
    def __init__(self, bs, nm):
        self.basic_scaffs = bs
        self.node_mappings = nm
        self.conflict_graph = None
        self.start_time = -1
        self.end_time = -1

    def gen_conflict_graph(self):
        print("gen_conflict_graph: implement me")

class TMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

class HMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)
"""
InputScaff: The full input scaff
"""


class InputScaff:
    id_incr = 0
    def __init__(self, node_list, edge_list, pushing_direction):
        self.hinge_graph = None
        self.mid_scaffs = []

        #patch list
        self.node_list = node_list
        #refers to indices in patch list
        self.edge_list = edge_list
        #axis vec3
        self.pushing_direction = pushing_direction

        #self.node_list_type = []

        # debug purposes for ease of our test algorithm
        # for now we manually define basic scaffolds
        self.basic_scaffs = []

        #This is for mapping for basic scaffs ids (ints) to node ids (list of ints)
        self.basic_mappings = {}

        # Decomposes self and generates scaffolds
        self.gen_scaffs()

    def gen_scaffs(self):
        # TODO: Di
        # generates basic scaffolds
        # generates hinge graph
        # generates mid-level scaffolds

        self.gen_hinge_graph()

        self.gen_basic_scaffs()

        self.gen_mid_scaffs()

    def gen_hinge_graph(self):    
        print("gen_hinge_graph...")
        for patch in self.node_list:
            ang = abs(np.dot(normalize(self.pushing_direction), normalize(patch.normal)))
            if ang < .1:
                patch.patch_type = PatchType.Fold
            else:
                patch.patch_type = PatchType.Base

        self.hinge_graph = networkx.Graph()
        for patch in self.node_list:
            self.hinge_graph.add_node(patch.id)

        for edge in self.edge_list:
            self.hinge_graph.add_edge(edge[0], edge[1])
    
    def gen_basic_scaffs(self):
        for patch in self.node_list:
            if patch.patch_type == PatchType.Fold:
                id = patch.id
                neighbors = list(self.hinge_graph.neighbors(id))
                if len(neighbors) == 2:
                    base0 = self.node_list[neighbors[0]]
                    base1 = self.node_list[neighbors[1]]
                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(HBasicScaff(fold0, base0, base1))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id, self.node_list[neighbors[1]].id]
                elif len(neighbors) == 1:
                    base0 = self.node_list[neighbors[0]]
                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(TBasicScaff(fold0, base0))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id]
                else:
                    print("wtf")

    def gen_mid_scaffs(self):
        di_graph_rep = self.hinge_graph.to_directed()
        cycles = sorted(nx.simple_cycles(di_graph_rep))
        cycles_filtered = []
        for l in cycles:
            if len(l) > 2:
                cycles_filtered.append(l)

        # Tracker is used to see which basic scaffs have been assigned to mid level scaffolds
        # This may cause issues because not using generated ids
        tracker = [False for i in range(len(self.basic_scaffs))]

        # Iterate through all cycles
        for cycle in cycles_filtered:
            basic_scaff_id_list = []
            patch_id_list = []

            # A basic scaffold is part of a mid level scaffolds if it's part of a cycle
            for scaff_id, patch_ids in self.basic_mappings.items():
                reject = False
                patch_id_temp = []
                for id in patch_ids:
                    if id not in cycle:
                        reject = True
                        break
                    else:
                        patch_id_temp.append(id)
                if not reject:
                    basic_scaff_id_list.append(scaff_id)
                    tracker[scaff_id] = True
                    patch_id_list = patch_id_list + patch_id_temp
            pruned_id_list = [*set(patch_id_list)]
        
            basic_scaff_list = []
            for scaff_id in basic_scaff_id_list:
                basic_scaff_list.append(self.basic_scaffs[scaff_id])

            self.mid_scaffs.append(HMidScaff(basic_scaff_list, pruned_id_list))

        # For midlevels that just compose a basic scaff
        # for idx in range(0, len(tracker)):
        #     if not tracker[idx]:
        #         if len(self.basic_mappings[idx]) == 3:
        #             self.mid_scaffs.append(HMidScaff([self.basic_scaffs[idx]], self.basic_mappings[idx]))
        #         else:
        #             self.mid_scaffs.append(TMidScaff([self.basic_scaffs[idx]], self.basic_mappings[idx]))

            
'''
FoldManager: debug class for now, probalby won't actually use it.
Purpose is to serve as a mini inputScaffold for now.
'''

class FoldManager:
    def __init__(self):
        self.h_basic_scaff = None # For now a hard coded H scaffold

    def generate_h_basic_scaff(self, bottom_patch: list, fold_patch: list, top_patch: list):
        print("generate_h_basic_scaff...")
        print("bottom patch")
        bPatch = Patch(bottom_patch)
        print("top patch")
        tPatch = Patch(top_patch)
        print("fold patch")
        fPatch = Patch(fold_patch)

        scaff = HBasicScaff(fPatch, bPatch, tPatch)
        self.h_basic_scaff = scaff

    def mainFold(self, nH) -> FoldOption:
        print("entered mainFold...")
        # Outputs a hard coded fold option for now

        # Experiment with alpha values
        alpha = 0.5
        cost1 = alpha * 0 / 1 + (1 - alpha) / 1
        mod1 = Modification(nH, 0, 1, 1, cost1)
        patch_list = [self.h_basic_scaff.f_patch, self.h_basic_scaff.b_patch_low, self.h_basic_scaff.b_patch_high]
        fo = FoldOption(True, mod1, patch_list)
        fo.gen_fold_transform()

        return fo



def basic_t_scaffold():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)]) # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)]) # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)]) # foldable patch

    foldable = Patch(coords3)
    base = Patch(coords2)
    tscaff = TBasicScaff(foldable, base)
    tscaff.gen_fold_options(1, 1, .5)
    print("Begin test")

    for scaff in tscaff.fold_options:
        for start in scaff.fold_transform.startAngles:
            print(start)
        for end in scaff.fold_transform.endAngles:
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
    tscaff.gen_fold_options(4, 1, .5)
    print("Begin test")

    for scaff in tscaff.fold_options:
        # for start in scaff.fold_transform.startAngles:
        #     print(start)
        # for end in scaff.fold_transform.endAngles:
        #     print(end)
        print(scaff.modification.range_start)
        print(scaff.modification.range_end)
        print(scaff.modification.num_pieces)
        print('------------------------------')

def basic_input_scaff():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)]) # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)]) # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)]) # foldable patch
    coords4 = np.array([(0, 2, 2), (0, 4, 2), (0, 4, 0), (0, 2, 0)])

    f1 = Patch(coords3)
    b1 = Patch(coords2)
    b2 = Patch(coords1)
    f2 = Patch(coords4)

    input = InputScaff([b1, f1, b2, f2], [[0,1],[1,2],[2,3]], normalize([0, 1, 0]))

    input.gen_hinge_graph()
    print(input.hinge_graph)
# basic_t_scaffold()
# basic_h_scaffold()

basic_input_scaff()

