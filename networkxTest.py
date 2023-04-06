from __future__ import annotations
import networkx as nx
import numpy as np
from enum import Enum
from typing import Dict, List, Set


class Axis(Enum):
    X = [1, 0, 0]
    Y = [0, 1, 0]
    Z = [0, 0, 1]


class PatchType(Enum):
    Base = 0
    Fold = 1


"""
Static helper functions for linear algebra
"""


def normalize(vec1):
    len = np.linalg.norm(vec1)
    return vec1 / len


# TODO: Might not even need this for now
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


def rectangleOverlap(rect1: List[List[float]], rect2: List[List[float]]) -> bool:
    rect1_minX = min(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
    rect1_maxX = max(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
    rect1_minY = min(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
    rect1_maxY = max(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])

    # if any vertex lies within the rectangle above, then there is some degree of overlap
    for vertex in rect2:
        if rect1_minX <= vertex[0] <= rect1_maxX and rect1_minY <= vertex[1] <= rect1_maxY:
            return True

    return False


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

        # numpy arrays
        self.coords: np.ndarray = rect_coords

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

        # start of range
        self.range_start = range_start
        # end of range
        self.range_end = range_end
        # number of pieces
        self.num_pieces = num_pieces

        # Hinge variables
        self.num_hinges = num_hinges


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

        self.startTime = 0
        self.endTime = 90


"""
FoldOption: An object that contains a modification and the associated fold transform
"""


class FoldOption:
    def __init__(self, isleft: bool, mod: Modification, scaff: BasicScaff):
        self.modification: Modification = mod
        self.isleft: bool = isleft
        self.fold_transform: FoldTransform = None

        # this is the id of the scaffold that this fold option is associated with
        self.scaff: BasicScaff = scaff

        # Traversal vector of the modification
        self.patch_traversal: List[float] = []

        # Projected region of the modification, a list of vertices in world space
        self.projected_region: List[float] = []

    def gen_fold_transform(self):
        print("Entered gen_fold_transform...")
        # Based on the modification, generate start angle and end angle
        # TODO: Di or David

        num_hinges = self.modification.num_hinges

        start_config = []  # FoldConfig()
        end_config = []  # FoldConfig()

        # When num_hinges is 0, it means we still have the hinge at the base patch(es)
        # meaning we always have 1 more than we expect
        for i in range(0, num_hinges + 1):

            start_ang = 0.0
            end_and = 0.0
            # first base patch and foldable patch connection
            # Will always be a 90 degree rotation.
            if i is 0:
                start_ang = 0.0
                end_ang = 90.0

            # final hinge, different based on if t or h
            # In the test case it should not even be hitting these
            # even hinges
            elif i % 2 is 0:
                start_ang = 0.0
                end_ang = 180.0
            # odd hinges
            else:
                start_ang = 0.0
                end_ang = -180.0

            if not self.isleft:
                start_ang *= -1
                end_and *= -1

            # Add the angles to the start and end configs
            start_config.append(start_ang)
            end_config.append(end_ang)

        # If isHScaff is True, then we need to add the final hinge
        # if isHScaff:
        #     start_config.append(0.0)
        #     end_config.append(-90.0)

        self.fold_transform = FoldTransform(start_config, end_config)

    def conflicts_with(self, other: FoldOption) -> bool:
        print("Checking conflict relationship...")
        if (other.scaff.id == self.scaff.id):
            # If they belong to the same scaffold, return true
            return True
        elif (other.fold_transform.startTime > self.fold_transform.endTime or
              other.fold_transform.endTime < self.fold_transform.startTime):
            # If they don't overlap in time, return false
            return False
        else:
            # If they overlap in time, then check if they share any of the same base patches
            # They should both be H scaffolds
            if (type(self.scaff) == HBasicScaff and type(other.scaff) == HBasicScaff):
                if (self.scaff.b_patch_low.id == other.scaff.b_patch_low.id and
                        self.scaff.b_patch_high.id == other.scaff.b_patch_high.id):

                    # Check their patch traversal trajectory is the same.
                    # If they are the same, then they are not conflicting
                    # TODO: If fold scaffs are actually really close to each other, then they should allow slanted folding STRETCH GOAL
                    # if (self.modification.patch_traversal == other.modification.patch_traversal):
                    #     return False

                    # I think, if they share the base patch and if they have an EVEN number of hinges
                    # UNLESS the number of hinges are the same, and fold direction is the same
                    # then they are always conflicting bc trajectory is not the same
                    if (self.modification.num_hinges % 2 == 0 and other.modification.num_hinges % 2 == 0
                            and self.isleft == other.isleft):
                        if (self.modification.num_hinges == other.modification.num_hinges):
                            return True
                        else:
                            return False

                    # if odd number of patches, trajectory will always be down and then move on.

                elif (self.scaff.b_patch_high.id == other.scaff.b_patch_high.id):
                    # Special case, if the scaffolds share the same high base patch,
                    # then their start time should be the same
                    if (self.fold_transform.startTime != other.fold_transform.startTime):
                        return True

                # Check their projected foldable areas for the foldable patches don't overlap
                self_region_vertices = self.projected_region
                other_region_vertices = other.projected_region
                if (rectangleOverlap(self_region_vertices, other_region_vertices)):
                    return True
                return False

            else:
                raise Exception("FoldOption::conflicts_with: Not both HBasicScaff")

    def calc_projected_region(self, rotAxis: List[float]) -> List[List[float]]:
        # calculates the projected region of the patch after a modification is applied
        # returns a list of 4 points

        # If left fold solution, then subtract points from current middle point
        # If right fold solution, then add points to current middle point

        f_patch = self.scaff.f_patch

        pMinX = min(f_patch.coords[0][0], f_patch.coords[1][0], f_patch.coords[2][0], f_patch.coords[3][0])
        pMaxX = max(f_patch.coords[0][0], f_patch.coords[1][0], f_patch.coords[2][0], f_patch.coords[3][0])
        pMinY = min(f_patch.coords[0][1], f_patch.coords[1][1], f_patch.coords[2][1], f_patch.coords[3][1])
        pMaxY = max(f_patch.coords[0][1], f_patch.coords[1][1], f_patch.coords[2][1], f_patch.coords[3][1])
        pMinZ = min(f_patch.coords[0][2], f_patch.coords[1][2], f_patch.coords[2][2], f_patch.coords[3][2])
        pMaxZ = max(f_patch.coords[0][2], f_patch.coords[1][2], f_patch.coords[2][2], f_patch.coords[3][2])

        norm = f_patch.calc_normal()

        if (norm == Axis.X):
            # in the y z plane
            if (rotAxis == Axis.Y):
                patchWidth = pMaxY - pMinY
                patchLength = pMaxZ - pMinZ

                # Obtain its "bottom" two vertices
                bottomVerts = [[pMaxX, pMaxY, pMinZ], [pMaxX, pMinY, pMinZ]]
            elif (rotAxis == Axis.Z):
                patchWidth = pMaxZ - pMinZ
                patchLength = pMaxY - pMinY

                # Obtain its "bottom" two vertices
                bottomVerts = [[pMaxX, pMinY, pMinZ], [pMaxX, pMinY, pMaxZ]]
            else:
                raise Exception("Invalid rotation axis... Returning")

        if (norm == Axis.Y):
            # in the x z plane
            if (rotAxis == Axis.X):
                patchWidth = pMaxX - pMinX
                patchLength = pMaxZ - pMinZ

                # obtain two bottom vertices
                bottomVerts = [[pMinX, pMaxY, pMinZ], [pMaxX, pMaxY, pMinZ]]

            elif (rotAxis == Axis.Z):
                patchWidth = pMaxZ - pMinZ
                patchLength = pMaxX - pMinX

                # obtain two bottom vertices
                bottomVerts = [[pMinX, pMaxY, pMinZ], [pMinX, pMaxY, pMaxZ]]

            else:
                raise Exception("Invalid rotation axis... Returning")

        if (norm == Axis.Z):
            # in the x y plane
            if (rotAxis == Axis.X):
                patchWidth = pMaxX - pMinX
                patchLength = pMaxY - pMinY

                # obtain two bottom vertices
                bottomVerts = [[pMinX, pMaxY, pMinZ], [pMaxX, pMaxY, pMinZ]]
            elif (rotAxis == Axis.Y):
                patchWidth = pMaxY - pMinY
                patchLength = pMaxX - pMinX

                # obtain two bottom vertices
                bottomVerts = [[pMinX, pMaxY, pMinZ], [pMinX, pMinY, pMinZ]]
            else:
                raise Exception("Invalid rotation axis... Returning")

        # Compute final length based on number of hinges
        finalLength = patchLength / (self.modification.num_hinges + 1)

        # Compute final width based on shrink value
        finalWidth = patchWidth / self.modification.num_pieces

        # compute the bottom vertices based on finalWidth and range_start and range_end
        # TODO: for now, hard code place to shrink from as the first element of bottomVerts
        newBottomVerts = []
        newBottomVerts[0] = bottomVerts[0] + self.modification.range_start * finalWidth
        newBottomVerts[1] = bottomVerts[0] + self.modification.range_end * finalWidth

        # Get base patch location
        newBottomVerts = np.array(newBottomVerts)

        # if Solution is left
        if (self.isleft):
            # Additional verts is on the left
            additionalVerts = newBottomVerts - finalLength
        else:
            # Additional verts is on the right
            additionalVerts = newBottomVerts + finalLength

        return [newBottomVerts[0], newBottomVerts[1], additionalVerts[1], additionalVerts[0]]


class BasicScaff():
    id_incr = 0

    def __init__(self):
        self.fold_options = []
        BasicScaff.id_incr += 1
        self.id = BasicScaff.id_incr

        # To be filled by the MWISP
        self.optimal_fold_option = None


class TBasicScaff(BasicScaff):
    def __init__(self, b_patch, f_patch):
        super().__init__()
        self.f_patch = f_patch
        self.b_patch = b_patch
        self.rot_axis = np.cross(f_patch.calc_normal(), b_patch.calc_normal())

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        patch_list = [self.b_patch, self.f_patch]
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                for k in range(0, j):
                    cost = alpha * i / nh + (1 - alpha) / ns

                    # TODO: Fix the modification to include patch shrinking?
                    mod = Modification(i, j / ns, k / ns, j - k, cost)

                    # compute the new vertex locations for the foldablke patch as its projected fold region
                    new_verts = self.f_patch.calc_projected_region(mod)

                    fo_left = FoldOption(True, mod, self)
                    fo_right = FoldOption(False, mod, self)

                    # generate the fold transform from start to end?
                    fo_left.gen_fold_transform()
                    fo_right.gen_fold_transform()

                    self.fold_options.append(fo_left)
                    self.fold_options.append(fo_right)


class HBasicScaff(BasicScaff):
    def __init__(self, b_patch_low, f_patch, b_patch_high):
        super().__init__()
        self.f_patch = f_patch
        self.b_patch_low = b_patch_low
        self.b_patch_high = b_patch_high
        self.rot_axis = np.cross(f_patch.calc_normal(), b_patch_low.calc_normal())

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        patch_list = [self.b_patch_low, self.f_patch, self.b_patch_high]
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                for k in range(0, j):
                    print(k)
                    print(j)
                    print("end")
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j / ns, k / ns, j - k, cost)
                    fo_left = FoldOption(True, mod, self)
                    fo_right = FoldOption(False, mod, self)

                    # generate the fold transform from start to end?
                    fo_left.gen_fold_transform()
                    fo_right.gen_fold_transform()

                    self.fold_options.append(fo_left)
                    self.fold_options.append(fo_right)


class MidScaff:
    def __init__(self, bs, nm):
        self.basic_scaffs: List[BasicScaff] = bs
        self.node_mappings = nm
        self.conflict_graph = None
        self.start_time = -1
        self.end_time = -1

    def gen_conflict_graph(self):
        print("generating conflict graph...")

        # We are generating the complement of the conflict graph, actually
        # self.conflict_graph = nx.Graph()

        # Assuming at this point we havea  list of basic scaffolds
        # For each scaffold, get its fold options and add them as nodes to the conflict grpah
        nodes = []
        node_weights = {}
        for scaff in self.basic_scaffs:
            for option in scaff.fold_options:
                nodes.append(option)
                node_weights[option] = option.modification.cost

        self.conflict_graph = nx.complete_graph(nodes)
        nx.set_node_attributes(self.conflict_graph, node_weights, "weight")

        # Now add the edges between each node
        # All nodes should be of the FoldOption type
        for option in self.conflict_graph.nodes:
            for other_option in self.conflict_graph.nodes:
                # if the two options are not the same and their relationship hasn't been evaluated yet
                if option != other_option and self.conflict_graph.has_edge(option, other_option) == False:
                    # Check if the two nodes conflict
                    if not option.conflicts_with(other_option):
                        # if they conflict, remove this edge
                        self.conflict_graph.remove_edge(option, other_option)

    def run_mwisp(self):
        print("running MWISP")

        # Run MWISP on the conflict graph
        # This will return a list of nodes (FoldOptions) that are in the maximum weight independent set
        # TODO: this might be really slow.
        max_clique = nx.algorithms.approximation.clique.max_weight_clique(self.conflict_graph, weight="weight")

        for fold_option in max_clique:
            # TODO: clean this up messy.
            fold_option.basic_scaff.optimal_fold_option = fold_option


class TMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)


class HMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)


class InputScaff:
    id_incr = 0

    def __init__(self, node_list, edge_list, pushing_direction):
        self.hinge_graph = None
        self.mid_scaffs = []

        # patch list
        self.node_list = node_list
        # refers to indices in patch list
        self.edge_list = edge_list
        # axis vec3
        self.pushing_direction = pushing_direction

        # self.node_list_type = []

        # debug purposes for ease of our test algorithm
        # for now we manually define basic scaffolds
        self.basic_scaffs = []

        # This is for mapping for basic scaffs ids (ints) to node ids (list of ints)
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

        self.hinge_graph = nx.Graph()
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
                    self.basic_scaffs.append(HBasicScaff(base0, fold0, base1))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id,
                                                                     self.node_list[neighbors[1]].id]
                elif len(neighbors) == 1:
                    base0 = self.node_list[neighbors[0]]
                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(TBasicScaff(base0, fold0))
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


# NETWORKX TESTING
class dummy:
    def __init__(self, id):
        self.id = id


d1 = dummy(1)
d2 = dummy(2)
d3 = dummy(3)
d4 = dummy(4)
d5 = dummy(5)
d6 = dummy(6)

G = nx.Graph()
Gnodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# make dictionary of weights corresponding to each node
Gweights = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
            6: 1, 7: 1, 8: 1, 9: 1}
# G.add_node(1, weight=1)
# G.add_node(2, weight=2)
# G.add_node(3, weight=1)
# G.add_node(4, weight=1)
# G.add_node(5, weight=1)
# G.add_node(6, weight=1)
# G.add_node(7, weight=1)
# G.add_node(8, weight=1)
# G.add_node(9, weight=1)

G.add_nodes_from(Gnodes)
nx.set_node_attributes(G, Gweights, 'weight')
print(G.nodes.data('weight'))

# G.add_edges_from([(1,3), (1, 4), (1, 5), (4, 2), (5, 2)])
# G.add_edges_from([(2, 3), (3, 4), (4, 5), (5, 6), (6, 2),
#                   (2, 4), (2, 5), (3, 5), (3, 6), (4, 6)])
# G.add_edges_from([(d2, d3), (d3, d4), (d4, d5), (d5, d6), (d6, d2),
#                   (d2, d4), (d2, d5), (d3, d5), (d3, d6), (d4, d6)])
# G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 3)])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 3),
                  (2, 4), (4, 6), (6, 7), (7, 8), (8, 9), (9, 6),
                  (6, 8), (9, 7)])

GComp = nx.complement(G)
nx.set_node_attributes(GComp, Gweights, "weight")

clique = nx.algorithms.clique.max_weight_clique(GComp, weight="weight")
print(clique)


