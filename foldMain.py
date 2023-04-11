# Unhinged main.py
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

def rectangleArea(rect: List[List[float]]):
    # calculate the area of the rectangle
    h = np.linalg.norm(rect[1] - rect[0])
    w = np.linalg.norm(rect[3] - rect[0])

    return h * w

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
        self.id = Patch.id_incr
        Patch.id_incr += 1

        # patch type:
        self.patch_type = None

    def calc_area(self):
        # calculates the area of the patch
        return rectangleArea(self.coords)

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

    #     # region
    #     self.projected_region = self.calc_region()
    #
    # def calc_region(self):
    #     print("calc_region: implement me")
    #     return -1


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

        # TODO: maybe move elsewhere
        self.starTime = 0
        self.endTime = 90


"""
FoldOption: An object that contains a modification and the associated fold transform
"""


class FoldOption:
    def __init__(self, isleft, mod: Modification, scaff: BasicScaff):
        self.modification: Modification = mod
        self.isleft: bool = isleft
        self.fold_transform: FoldTransform = None
        # self.rot_axis = []

        # this patch list should be at least size 2
        # self.patch_list = patch_list

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

        # crossing the normals of base patch and normal patch
        # self.rot_axis = np.cross(self.patch_list[1].calc_normal(), self.patch_list[0].calc_normal())

        # using patch_list to determine t or h scaffold
        # num_hinges = self.modification.numSplits + len(self.patch_list) - 1
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
                end_ang = 90.0
            # odd hinges
            else:
                start_ang = 0.0
                end_ang = -90.0

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

    def get_patch_width_length_bottom(self, norm, rotAxis, minX, maxX, minY, maxY, minZ, maxZ):
        params = {
            Axis.X: {
                Axis.Y: {
                    'patch_width': maxY - minY,
                    'patch_length': maxZ - minZ,
                    'bottom_verts': [[maxX, maxY, minZ], [maxX, minY, minZ]]
                },
                Axis.Z: {
                    'patch_width': maxZ - minZ,
                    'patch_length': maxY - minY,
                    'bottom_verts': [[maxX, minY, minZ], [maxX, minY, maxZ]]
                }
            },
            Axis.Y: {
                Axis.X: {
                    'patch_width': maxX - minX,
                    'patch_length': maxZ - minZ,
                    'bottom_verts': [[minX, maxY, minZ], [maxX, maxY, minZ]]
                },
                Axis.Z: {
                    'patch_width': maxZ - minZ,
                    'patch_length': maxX - minX,
                    'bottom_verts': [[minX, maxY, minZ], [minX, maxY, maxZ]]
                }
            },
            Axis.Z: {
                Axis.X: {
                    'patch_width': maxX - minX,
                    'patch_length': maxY - minY,
                    'bottom_verts': [[minX, maxY, minZ], [maxX, maxY, minZ]]
                },
                Axis.Y: {
                    'patch_width': maxY - minY,
                    'patch_length': maxX - minX,
                    'bottom_verts': [[minX, maxY, minZ], [minX, minY, minZ]]
                }
            }
        }
        return params.get(norm, {}).get(rotAxis, None)

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

        result = self.get_patch_width_length_bottom(norm, rotAxis, pMinX, pMaxX, pMinY, pMaxY, pMinZ, pMaxZ)

        if result is None:
            raise Exception("Invalid rotation axis... Returning")

        patchWidth = result['patch_width']
        patchLength = result['patch_length']
        bottomVerts = result['bottom_verts']

        print("PatchWidth: " + str(patchWidth))
        print("PatchLength: " + str(patchLength))
        print("BottomVerts: ")
        print(bottomVerts)

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


"""
BasicScaff: Parent class for TBasicScaff and HBasicScaff
"""


class BasicScaff():
    id_incr = 0

    def __init__(self):
        self.fold_options = []
        self.id = BasicScaff.id_incr
        BasicScaff.id_incr += 1

        # To be filled by the MWISP
        self.optimal_fold_option = None


"""
TBasicScaff: A basic scaffold of type T
"""


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
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                for k in range(0, j):
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j, k, j - k, cost)
                    # fo_left = FoldOption(True, mod, patch_list)
                    # fo_right = FoldOption(False, mod, patch_list)

                    fo_left = FoldOption(True, mod, self)
                    fo_right = FoldOption(False, mod, self)

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

        self.rot_axis = np.cross(f_patch.calc_normal(), b_patch_low.calc_normal())

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        for i in range(0, nh + 1):
            for j in range(0, ns + 1):
                for k in range(0, j):
                    print(k)
                    print(j)
                    print("end")
                    cost = alpha * i / nh + (1 - alpha) / ns
                    mod = Modification(i, j, k, j - k, cost)
                    # fo_left = FoldOption(True, mod, patch_list)
                    # fo_right = FoldOption(False, mod, patch_list)

                    fo_left = FoldOption(True, mod, self)
                    fo_right = FoldOption(False, mod, self)

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

class TMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

    # Only one solution? Is there any case where this solution could impact another guy?
    # Probably not since they all fold at different times? Sequentially?

class HMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

    def gen_conflict_graph(self):
        print("generating conflict graph...")

        # We are generating the complement of the conflict graph, actually
        # self.conflict_graph = nx.Graph()

        # Assuming at this point we havea  list of basic scaffolds
        # For each scaffold, get its fold options and add them as nodes to the conflict grpah
        nodes = []
        node_weights = {}

        # Sum of all the folded areas
        sum_fold_area = 0

        # Cost of the most expensive solution
        max_cost_v = -1

        for scaff in self.basic_scaffs:
            # Sum of the final area of every modification
            for option in scaff.fold_options:
                region_area = rectangleArea(option.projected_region)
                sum_fold_area += region_area
                max_cost_v = max(max_cost_v, option.modification.cost)

        for scaff in self.basic_scaffs:
            for option in scaff.fold_options:
                region_area = rectangleArea(option.projected_region)
                lambda_i = region_area / sum_fold_area
                cost_vj = lambda_i * option.modification.cost
                weight = max_cost_v - cost_vj + 1

                nodes.append(option)
                node_weights[option] = weight

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

    def fold(self):
        self.gen_conflict_graph()
        self.run_mwisp()


"""
InputScaff: The full input scaff
"""


class InputScaff:
    id_incr = 0

    def __init__(self, node_list, edge_list, push_dir, max_hinges, num_shrinks):
        self.hinge_graph = None  # Gets created gen_hinge_graph
        self.mid_scaffs = []

        # patch list
        self.node_list = node_list
        # refers to indices in patch list
        self.edge_list = edge_list
        # axis vec3
        if (push_dir == [0, 1, 0]):
            self.push_dir: Axis = Axis.Y
        elif (push_dir == [0, 0, 1]):
            self.push_dir: Axis = Axis.Z
        elif (push_dir == [1, 0, 0]):
            self.push_dir: Axis = Axis.X
        else:
            raise Exception("Invalid push direction")
            exit (-1)


        # debug purposes for ease of our test algorithm
        # for now we manually define basic scaffolds
        self.basic_scaffs = []

        # This is for mapping for basic scaffs ids (ints) to node ids (list of ints)
        self.basic_mappings = {}

        self.max_hinges = max_hinges
        self.num_shrinks = num_shrinks

        # Decomposes self and generates scaffolds
        # TODO: commented out for testing purposes, recomment back in
        # self.gen_scaffs()

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
            ang = abs(np.dot(normalize(self.push_dir), normalize(patch.normal)))
            print(normalize(patch.normal))
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
        print("gen basic scaffs")
        for patch in self.node_list:
            if patch.patch_type == PatchType.Fold:
                id: int = patch.id
                neighbors = list(self.hinge_graph.neighbors(id))
                if len(neighbors) == 2:
                    # TODO: Always assume push axis is negative for now

                    # If push axis is negative, base_hi is the one with higher value along the pos of push axis
                    base1: Patch = self.node_list[neighbors[0]]
                    base2: Patch = self.node_list[neighbors[1]]
                    if self.push_dir == Axis.X:
                        if (base1.coords[0] > base2.coords[0]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    elif self.push_dir == Axis.Y:
                        if (base1.coords[1] > base2.coords[1]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    elif self.push_dir == Axis.Z:
                        if (base1.coords[2] > base2.coords[2]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    else:
                        raise Exception("Invalid push direction")
                        exit(-1)

                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(HBasicScaff(base_lo, fold0, base_hi))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id,
                                                                     self.node_list[neighbors[1]].id]
                    print(self.basic_scaffs[-1].id)
                elif len(neighbors) == 1:
                    base0 = self.node_list[neighbors[0]]
                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(TBasicScaff(base0, fold0))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id]
                    print(self.basic_scaffs[-1].id)
                else:
                    print("wtf, no neighbors in the hinge graph??: " + str(id))
        print("end gen basic scaffs")

    # Basically just removes non-unique lists
    def remove_duplicate_cycles(self, cycle_list):
        cycles_unique = []

        for cycle in cycle_list:
            is_unique = True
            for u_cycle in cycles_unique:
                c1 = u_cycle.copy()
                c2 = cycle.copy()
                c1.sort()
                c2.sort()
                if c1 == c2:
                    is_unique = False
                    break
            if is_unique:
                cycles_unique.append(cycle)
        return cycles_unique

    # Cycle merging is for combining all the unique nodes of cycles that share foldable patches
    # This is necessary for creating mid level scaffolds
    def merge_cycles(self, cycle_list):
        merged_cycles = []
        tracker = [False for i in range(len(cycle_list))]
        for i in range(0, len(cycle_list)):
            # Check only unmerged cycles
            if not tracker[i]:
                merged_cycles.append(cycle_list[i])
                tracker[i] = True
                working_cycle_id = len(merged_cycles) - 1

                # Check every cycle to see if it can be merged to current one
                for j in range(0, len(cycle_list)):
                    # Again only check if not yet merged
                    if not tracker[j]:
                        # Iterate through elements
                        for id in cycle_list[j]:
                            # Merge cycles if the node exists in the first cycle and the node is a foldable patch
                            if id in merged_cycles[working_cycle_id] and self.node_list[
                                id].patch_type == PatchType.Fold:
                                # Append all unique nodes to new cycle
                                tracker[j] = True
                                for id2 in cycle_list[j]:
                                    if id2 not in merged_cycles[working_cycle_id]:
                                        merged_cycles[working_cycle_id].append(id2)
        return merged_cycles

    def gen_mid_scaffs(self):
        di_graph_rep = self.hinge_graph.to_directed()
        cycles = sorted(nx.simple_cycles(di_graph_rep))
        cycles_big = []
        for l in cycles:
            if len(l) > 2:
                cycles_big.append(l)

        cycles_filtered = self.merge_cycles(self.remove_duplicate_cycles(cycles_big))

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
        # print(self.basic_mappings)
        for idx in range(0, len(tracker)):
            # print(idx)
            if not tracker[idx]:
                if len(self.basic_mappings[idx]) == 3:
                    self.mid_scaffs.append(HMidScaff([self.basic_scaffs[idx]], self.basic_mappings[idx]))
                else:
                    self.mid_scaffs.append(TMidScaff([self.basic_scaffs[idx]], self.basic_mappings[idx]))

    def fold(self):
        # TODO: For now, only foldabilize the first mid level scaffold.
        # TODO: Eventually I will need to do greedy one step lookahead.

        # First, generate basic scaffold solutions
        for scaff in self.basic_scaffs:
            scaff.gen_fold_options()

        self.mid_scaffs[0].fold()

'''
FoldManager: debug class for now, probalby won't actually use it.
Purpose is to serve as a mini inputScaffold for now.
'''

# class FoldManager:
#     def __init__(self):
#         self.h_basic_scaff = None  # TODO: For now a hard coded H scaffold
#         self.input_scaff = None
#
#     def generate_h_basic_scaff(self, bottom_patch: list, fold_patch: list, top_patch: list):
#         print("generate_h_basic_scaff...")
#         print("bottom patch")
#         bPatch = Patch(bottom_patch)
#         print("top patch")
#         tPatch = Patch(top_patch)
#         print("fold patch")
#         fPatch = Patch(fold_patch)
#
#         scaff = HBasicScaff(fPatch, bPatch, tPatch)
#         self.h_basic_scaff = scaff
#
#     def mainFold(self, nH, nS) -> FoldOption:
#         print("entered mainFold...")
#         # Outputs a hard coded fold option for now
#
#         # Experiment with alpha values
#         alpha = 0.5
#         cost1 = alpha * 0 / 1 + (1 - alpha) / 1
#         mod1 = Modification(nH, 0, 3, nS, cost1)
#         patch_list = [self.h_basic_scaff.f_patch, self.h_basic_scaff.b_patch_low, self.h_basic_scaff.b_patch_high]
#         fo = FoldOption(False, mod1, patch_list)
#         fo.gen_fold_transform()
#
#         # TODO: hard coded for now
#         fo.fold_transform.startTime = 0
#         fo.fold_transform.endTime = 90
#
#         return fo


def basic_t_scaffold():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)])  # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)])  # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)])  # foldable patch

    foldable = Patch(coords3)
    base = Patch(coords2)
    tscaff = TBasicScaff(base, foldable)
    tscaff.gen_fold_options(1, 1, .5)
    print("Begin test")

    for scaff in tscaff.fold_options:
        for start in scaff.fold_transform.startAngles:
            print(start)
        for end in scaff.fold_transform.endAngles:
            print(end)
        print('------------------------------')


def basic_h_scaffold():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)])  # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)])  # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)])  # foldable patch

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
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)])  # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)])  # bottom base patch
    coords3 = np.array([(0, 0, 2), (0, 2, 2), (0, 2, 0), (0, 0, 0)])  # foldable patch
    coords4 = np.array([(0, 2, 2), (0, 4, 2), (0, 4, 0), (0, 2, 0)])

    b1 = Patch(coords2)
    f1 = Patch(coords3)
    b2 = Patch(coords1)
    f2 = Patch(coords4)

    b1.id = 0
    f1.id = 1
    b2.id = 2
    f2.id = 3

    input = InputScaff([b1, f1, b2, f2], [[0, 1], [1, 2], [2, 3]], normalize([0, 1, 0]))

    input.gen_hinge_graph()

    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")
    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)
    print("MIDSCAFFS")
    input.gen_mid_scaffs()
    print(input.mid_scaffs)
    for mid_scaff in input.mid_scaffs:
        print(mid_scaff.node_mappings)


# basic_input_scaff()

def interm1_input_scaff():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)])  # top base patch
    coords2 = np.array([(-1, 1, 2), (-1, 1, 0), (0, 1, 0), (0, 1, 2)])  # mid base patch
    coords3 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)])  # bottom base patch
    coords4 = np.array([(-1, 0, 2), (-1, 1, 0), (-1, 0, 0), (-1, 1, 2)])  # left bot fold patch
    coords5 = np.array([(-1, 1, 2), (-1, 2, 0), (-1, 1, 0), (-1, 2, 2)])  # left top fold
    coords6 = np.array([(1, 0, 2), (1, 2, 0), (1, 0, 0), (1, 2, 2)])  # right fold

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)
    f1 = Patch(coords4)
    f2 = Patch(coords5)
    f3 = Patch(coords6)

    b1.id = 0
    b2.id = 1
    b3.id = 2
    f1.id = 3
    f2.id = 4
    f3.id = 5

    input = InputScaff([b1, b2, b3, f1, f2, f3], [[0, 3], [0, 5], [1, 3], [1, 4], [2, 4], [2, 5]], normalize([0, 1, 0]), 2, 1)

    input.gen_hinge_graph()

    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")
    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)

    print("Remove Duplicate")
    print(input.remove_duplicate_cycles([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [0, 1, 2], [2, 3, 4]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [2, 1, 0], [2, 3, 4]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [2, 1, 0], [2, 3, 4], [2, 4, 3], [1, 3, 4]]))

    print("Merge cycles")
    print(input.merge_cycles([[0, 1, 2], [0, 1, 3]]))
    print(input.merge_cycles([[0, 1, 2], [0, 1, 3], [1, 2, 3]]))
    print(input.merge_cycles([[0, 3, 5], [0, 1, 3], [1, 2, 3], [0, 3, 4]]))

    print("MIDSCAFFS")
    input.gen_mid_scaffs()
    print(input.mid_scaffs)
    for mid_scaff in input.mid_scaffs:
        print(mid_scaff.node_mappings)


# interm1_input_scaff()

def interm2_input_scaff():
    coords1 = np.array([(-1, 2, 2), (-1, 2, 0), (1, 2, 0), (1, 2, 2)])  # top base patch
    coords2 = np.array([(-1, 0, 2), (-1, 0, 0), (1, 0, 0), (1, 0, 2)])  # bottom base patch
    coords3 = np.array([(-1, 0, 2), (-1, 2, 0), (-1, 0, 0), (-1, 2, 2)])  # left fold
    coords4 = np.array([(0, 0, 2), (0, 2, 0), (0, 0, 0), (0, 2, 2)])  # mid fold
    coords5 = np.array([(1, 0, 2), (1, 2, 0), (1, 0, 0), (1, 2, 2)])  # right fold

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)
    f2 = Patch(coords4)
    f3 = Patch(coords5)

    b1.id = 0
    b2.id = 1
    f1.id = 2
    f2.id = 3
    f3.id = 4

    input = InputScaff([b1, b2, f1, f2, f3], [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]], normalize([0, 1, 0]))

    input.gen_hinge_graph()

    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")
    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)
    print("MIDSCAFFS")
    input.gen_mid_scaffs()
    print(input.mid_scaffs)
    for mid_scaff in input.mid_scaffs:
        print(mid_scaff.node_mappings)

# interm2_input_scaff()

def test_conflict_graph():
    coords1 = np.array([(3, 2, 5.5), (4.5, 2, 5.5), (4.5, 2, 4.5), (3, 2, 4.5)])  # top base patch # 0
    coords2 = np.array([(3, 0, 5.5), (5, 0, 5.5), (5, 0, 4.5), (3, 0, 4.5)])  # bottom base patch # 1
    coords3 = np.array([(4, 1, 5.5), (5, 1, 5.5), (5, 1, 4.5), (4, 1, 4.5)])  # middle base patch # 2
    coords4 = np.array([(4.75, 0, 5.5), (4.75, 0, 4.5), (4.75, 1, 4.5), (4.75, 1, 5.5)])  # left bot fold patch # 3
    coords5 = np.array([(4.25, 1, 5.5), (4.25, 1, 4.5), (4.25, 2, 4.5), (4.25, 2, 5.5)])  # left top fold # 4
    coords6 = np.array([(3.25, 0, 5.5), (3.25, 0, 4.5), (3.25, 2, 4.5), (3.25, 3, 5.5)])  # right fold # 5

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)
    f1 = Patch(coords4)
    f2 = Patch(coords5)
    f3 = Patch(coords6)

    b1.id = 0
    b2.id = 1
    b3.id = 2

    f1.id = 3
    f2.id = 4
    f3.id = 5

    nodes = [b1, b2, b3, f1, f2, f3]
    edges = [[1, 3], [3, 2], [2, 4], [4, 0], [5, 0], [1, 5]]

    input = InputScaff(nodes, edges, normalize([0, -1, 0]), 2, 1)

    input.gen_hinge_graph()

    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")
    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)

    for basic_scaff in input.basic_scaffs:
        print("SCAFF =================== ")
        print("foldable: " + str(basic_scaff.f_patch.id))
        print("base high: " + str(basic_scaff.b_patch_high.id))
        print("base low: " + str(basic_scaff.b_patch_low.id))

    print("Remove Duplicate")
    print(input.remove_duplicate_cycles([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [0, 1, 2], [2, 3, 4]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [2, 1, 0], [2, 3, 4]]))
    print(input.remove_duplicate_cycles([[0, 1, 2], [2, 1, 0], [2, 3, 4], [2, 4, 3], [1, 3, 4]]))

    print("Merge cycles")
    print(input.merge_cycles([[0, 1, 2], [0, 1, 3]]))
    print(input.merge_cycles([[0, 1, 2], [0, 1, 3], [1, 2, 3]]))
    print(input.merge_cycles([[0, 3, 5], [0, 1, 3], [1, 2, 3], [0, 3, 4]]))

    print("MIDSCAFFS")
    input.gen_mid_scaffs()
    print(input.mid_scaffs)
    for mid_scaff in input.mid_scaffs:
        print(mid_scaff.node_mappings)


test_conflict_graph()
