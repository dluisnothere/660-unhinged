# Unhinged main.py
from __future__ import annotations
import networkx as nx
import numpy as np
from enum import Enum
from typing import Dict, List, Set

# Predefined constants

XAxis = np.array([-1, 0, 0])
YAxis = np.array([0, -1, 0])
ZAxis = np.array([0, 0, -1])

smoothbrainiqNumber = .00001
bigChungusNumber = 10000


class PatchType(Enum):
    Base = 0
    Fold = 1


"""
Static helper functions for linear algebra
"""


def normalize(vec1):
    length = np.linalg.norm(vec1)
    return vec1 / length


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


def calc_normal(rect: np.ndarray(np.ndarray(float))) -> np.ndarray(float):
    # calculates the normal of the patch
    # it can have two norms: and they're the negative of each other, but this function only returns one
    hvec = rect[1] - rect[0]
    wvec = rect[3] - rect[0]

    # Take the cross product, then normalize
    surf = np.cross(hvec, wvec)
    return normalize(surf)


def check_rectangle_overlap(rect1, rect2):
    # Find the minimum and maximum x and z values for both rectangles
    min_rect1_x, max_rect1_x = min(rect1[:, 0]), max(rect1[:, 0])
    min_rect1_z, max_rect1_z = min(rect1[:, 2]), max(rect1[:, 2])

    min_rect2_x, max_rect2_x = min(rect2[:, 0]), max(rect2[:, 0])
    min_rect2_z, max_rect2_z = min(rect2[:, 2]), max(rect2[:, 2])

    # Check if the rectangles overlap in the x and z dimensions
    x_overlap = min_rect1_x < max_rect2_x and min_rect2_x < max_rect1_x
    z_overlap = min_rect1_z < max_rect2_z and min_rect2_z < max_rect1_z

    # If both x and z dimensions overlap, the rectangles overlap
    return x_overlap and z_overlap

def rectangle_area(rect: np.ndarray(np.ndarray(float))):
    # calculate the area of the rectangle
    h = np.linalg.norm(rect[1] - rect[0])
    w = np.linalg.norm(rect[3] - rect[0])

    return h * w


"""
Patch: Our proxy for a rectangle and only contains a rectangle
"""


class Patch:
    id_incr = 0

    def __init__(self, rect_coords, name=''):
        # numpy arrays
        self.coords: np.ndarray = rect_coords

        # name of the patch (FOR MAYA NODE PURPOSES ONLY. OPTIONAL PARAM)
        self.name = name

        # a point on the plane, used for SDF
        self.constant = np.mean(rect_coords, axis=0)

        # calculate the normal of the surface immediately
        self.normal = calc_normal(self.coords)

        # calculate area
        self.area = rectangle_area(self.coords)

        # id for debug purposes
        self.id = Patch.id_incr
        Patch.id_incr += 1

        # patch type:
        self.patch_type = None

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
FoldTransform: An association of start and end angles with their associated time step
"""


class FoldTransform:
    def __init__(self, a_st, a_en, start_time, end_time):
        # FoldConfig
        self.startAngles = a_st
        # FoldConfig
        self.endAngles = a_en

        # TODO: maybe move elsewhere, or will become a float
        self.start_time = start_time
        self.end_time = end_time


"""
FoldOption: An object that contains a modification and the associated fold transform
"""


class FoldOption:
    def __init__(self, isleft, mod: Modification, scaff: BasicScaff, axis: np.ndarray(float)):
        self.modification: Modification = mod
        self.isleft: bool = isleft
        self.fold_transform: FoldTransform = None
        self.rot_axis = axis
        self.height = -1

        # this patch list should be at least size 2
        # self.patch_list = patch_list

        # this is the id of the scaffold that this fold option is associated with
        self.scaff: BasicScaff = scaff

        # Traversal vector of the modification
        self.patch_trajectory: np.ndarray(float) = None

        # Projected region of the modification, a list of vertices in world space
        self.projected_region: np.ndarray(np.ndarray(float)) = None

        # Box defined by minimum and maximum points
        self.bvh: np.ndarray(np.ndarray(float)) = None

        self.gen_fold_transform()
        self.gen_projected_region(self.rot_axis)
        self.construct_bvh(self.rot_axis)

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

        # print("scaff object: ")
        # print(self.scaff)
        # print("scaff start time: " + str(self.scaff.start_time))
        # print("scaff end time: " + str(self.scaff.end_time))

        if (self.scaff.start_time == -1 or self.scaff.end_time == -1):
            raise Exception("Error: start or end time not set for scaffold")
        print("Warning... start or end time not set yet")

        self.fold_transform = FoldTransform(start_config, end_config, self.scaff.start_time, self.scaff.end_time)
        # TODO: the time variables should be set by greedy one step lookahead, which hasn't happened yet

    def conflicts_with(self, other: FoldOption) -> bool:
        if (other.scaff.id == self.scaff.id):
            return True
        elif (other.fold_transform.start_time > self.fold_transform.end_time or
              other.fold_transform.end_time < self.fold_transform.start_time):
            # If they don't overlap in time, return false
            return False
        else:
            # If they overlap in time, then check if they share any of the same base patches
            # They should both be H scaffolds
            if (type(self.scaff) == HBasicScaff and type(other.scaff) == HBasicScaff):
                if (self.scaff.t_patch.id == other.scaff.t_patch.id):
                    # Special case, if the scaffolds share the same high base patch,
                    # then their start time should be the same
                    if (self.fold_transform.start_time != other.fold_transform.start_time):
                        return True
                    # if (self.modification.num_hinges % 2 != other.modification.num_hinges % 2):
                    #     # If one has even number and another has odd number hinges... immediately false Since they
                    #     # share the same top patch.
                    #     return True
                    # if (self.modification.num_hinges % 2 == 0 and other.modification.num_hinges % 2 == 0):
                    #     if (self.isleft != other.isleft):
                    #         return True
                    #     if (self.modification.num_hinges != other.modification.num_hinges):
                    #         return True
                elif (self.scaff.b_patch.id == other.scaff.b_patch.id):
                    if (self.fold_transform.end_time != other.fold_transform.end_time):
                        return True
                # STill need to check to make sure they are either both even or both odd.
                # Check their projected foldable areas for the foldable patches don't overlap
                self_region_vertices = self.projected_region
                other_region_vertices = other.projected_region
                if (check_rectangle_overlap(self_region_vertices, other_region_vertices)):
                    # print("Conflict between: " + str(self.scaff.id) + " and " + str(other.scaff.id))
                    # print("Times overlap and rectangle overlap detected!")
                    return True
                else:
                    return False

            else:
                raise Exception("FoldOption::conflicts_with: Not both HBasicScaff")

    def get_patch_width_length_bottom(self, rotAxis, b_patch: Patch, f_patch: Patch) -> (float, float, np.ndarray):
        f_norm = calc_normal(f_patch.coords)
        b_norm = calc_normal(b_patch.coords)

        # Get the maximum and minimum x,y,z of the foldable patch
        maxX = max(f_patch.coords[0][0], f_patch.coords[1][0], f_patch.coords[2][0], f_patch.coords[3][0])
        minX = min(f_patch.coords[0][0], f_patch.coords[1][0], f_patch.coords[2][0], f_patch.coords[3][0])
        maxY = max(f_patch.coords[0][1], f_patch.coords[1][1], f_patch.coords[2][1], f_patch.coords[3][1])
        minY = min(f_patch.coords[0][1], f_patch.coords[1][1], f_patch.coords[2][1], f_patch.coords[3][1])
        maxZ = max(f_patch.coords[0][2], f_patch.coords[1][2], f_patch.coords[2][2], f_patch.coords[3][2])
        minZ = min(f_patch.coords[0][2], f_patch.coords[1][2], f_patch.coords[2][2], f_patch.coords[3][2])

        # The f_patch edge that attaches to b_patch is the bottom edge, and also the width
        # The other edge is the height

        if (abs(np.dot(b_norm, XAxis)) == 1):
            if (abs(np.dot(f_norm, YAxis)) == 1):
                height = maxX - minX
                width = maxZ - minZ
                bottom_edge_verts = np.array([[b_patch.coords[0][0], f_patch.coords[0][1], maxZ],
                                              [b_patch.coords[0][0], f_patch.coords[0][1], minZ]])
            elif (abs(np.dot(f_norm, ZAxis)) == 1):
                height = maxX - minX
                width = maxY - minY
                bottom_edge_verts = np.array([[b_patch.coords[0][0], maxY, f_patch.coords[0][2]],
                                              [b_patch.coords[0][0], minY, f_patch.coords[0][2]]])
            else:
                # print("b_norm")
                # print(b_norm)
                # print("f_norm")
                # print(f_norm)
                raise Exception("FoldOption::get_patch_width_length_bottom: f_patch and b_patch share the same normal?")
        elif (abs(np.dot(b_norm, YAxis)) == 1):
            if (abs(np.dot(f_norm, XAxis)) == 1):
                height = maxY - minY
                width = maxZ - minZ
                bottom_edge_verts = np.array([[f_patch.coords[0][0], b_patch.coords[0][1], maxZ],
                                              [f_patch.coords[0][0], b_patch.coords[0][1], minZ]])
            elif (abs(np.dot(f_norm, ZAxis)) == 1):
                height = maxY - minY
                width = maxX - minX
                bottom_edge_verts = np.array([[maxX, b_patch.coords[0][1], f_patch.coords[0][2]],
                                              [minX, b_patch.coords[0][1], f_patch.coords[0][2]]])
            else:
                # print("b_norm")
                # print(b_norm)
                # print("f_norm")
                # print(f_norm)
                raise Exception("FoldOption::get_patch_width_length_bottom: f_patch and b_patch share the same normal?")
        elif (abs(np.dot(b_norm, ZAxis)) == 1):
            if (abs(np.dot(f_norm, XAxis)) == 1):
                height = maxZ - minZ
                width = maxY - minY
                bottom_edge_verts = np.array([[f_patch.coords[0][0], maxY, b_patch.coords[0][2]],
                                              [f_patch.coords[0][0], minY, b_patch.coords[0][2]]])
            elif (abs(np.dot(f_norm, YAxis)) == 1):
                height = maxZ - minZ
                width = maxX - minX
                bottom_edge_verts = np.array([[maxX, f_patch.coords[0][1], b_patch.coords[0][2]],
                                              [minX, f_patch.coords[0][1], b_patch.coords[0][2]]])

            else:
                # print("b_norm")
                # print(b_norm)
                # print("f_norm")
                # print(f_norm)
                raise Exception("FoldOption::get_patch_width_length_bottom: f_patch and b_patch share the same normal?")
        else:
            # print("b_norm")
            # print(b_norm)
            # print("f_norm")
            # print(f_norm)
            raise Exception("FoldOption::get_patch_width_length_bottom: patch normal incorrect?")

        return (width, height, bottom_edge_verts)

    def gen_projected_region(self, rotAxis: np.ndarray(float)):
        # calculates the projected region of the patch after a modification is applied
        # returns a list of 4 points

        # If left fold solution, then subtract points from current middle point
        # If right fold solution, then add points to current middle point

        f_patch = self.scaff.f_patch
        b_patch = self.scaff.b_patch

        patchWidth, patchHeight, bottomVerts = self.get_patch_width_length_bottom(rotAxis, b_patch, f_patch)

        self.height = patchHeight

        # print("patchWidth: ", patchWidth)
        # print("patchHeight: ", patchHeight)
        # print("bottomVerts: ", bottomVerts)

        # Compute final length based on number of hinges
        finalLength = patchHeight / (self.modification.num_hinges + 1)
        # Compute a translation vector based on finalLength in the direction of the normal of the foldable patch
        translationVec = finalLength * calc_normal(f_patch.coords)

        # print("translationVec: ", translationVec)

        # Compute final width based on shrink value
        finalWidth = patchWidth / self.modification.num_pieces

        # compute the bottom vertices based on finalWidth and range_start and range_end
        # TODO: for now, hard code place to shrink from as the first element of bottomVerts
        newBottomVerts = []

        # # TODO: debug print bottomVertis[0]
        # print("bottomVerts[0]: ", bottomVerts[0])
        # print("bottomVerts[1]: ", bottomVerts[1])

        # Compute a shrinkVec based on finalWidth in the direction of the rotationAxis
        shrinkVec0 = self.modification.range_start * finalWidth * np.abs(rotAxis)
        shrinkVec1 = self.modification.range_end * finalWidth * np.abs(rotAxis)

        # print("shrinkVec0: ", shrinkVec0)
        # print("shrinkVec1: ", shrinkVec1)

        if (abs(np.dot(rotAxis, XAxis)) == 1):
            if (bottomVerts[0][0] < bottomVerts[1][0]):
                startPos = bottomVerts[0]
            else:
                startPos = bottomVerts[1]
        elif (abs(np.dot(rotAxis, YAxis)) == 1):
            if (bottomVerts[0][1] < bottomVerts[1][1]):
                startPos = bottomVerts[0]
            else:
                startPos = bottomVerts[1]
        elif (abs(np.dot(rotAxis, ZAxis)) == 1):
            if (bottomVerts[0][2] < bottomVerts[1][2]):
                startPos = bottomVerts[0]
            else:
                startPos = bottomVerts[1]
        else:
            raise Exception("FoldOption::gen_projected_region: rotAxis is not a valid axis")

        newBottomVert0 = startPos + shrinkVec0
        # TODO: for now we are arbitrarily picking one of the bottom verts to shrink from
        newBottomVert1 = startPos + shrinkVec1

        # print("newCalcVert0: ", newBottomVert0)
        # print("newCalcVert1: ", newBottomVert1)

        newBottomVerts.append(newBottomVert0)
        newBottomVerts.append(newBottomVert1)

        # turn into a numpy array
        newBottomVerts = np.array(newBottomVerts)

        # if Solution is left
        if (self.isleft):
            # Additional verts is on the left
            additionalVerts: np.ndarray = newBottomVerts - translationVec
        else:
            # Additional verts is on the right
            additionalVerts: np.ndarray = newBottomVerts + translationVec

        self.projected_region = np.array([newBottomVerts[0], newBottomVerts[1], additionalVerts[1], additionalVerts[0]])

    def construct_bvh(self, rotAxis: np.ndarray(float)):
        bvh: list[list[float]] = []
        norm = normalize(calc_normal(self.projected_region))
        proj = self.projected_region

        maxX = max(proj[0][0], proj[1][0], proj[2][0], proj[3][0])
        minX = min(proj[0][0], proj[1][0], proj[2][0], proj[3][0])
        maxY = max(proj[0][1], proj[1][1], proj[2][1], proj[3][1])
        minY = min(proj[0][1], proj[1][1], proj[2][1], proj[3][1])
        maxZ = max(proj[0][2], proj[1][2], proj[2][2], proj[3][2])
        minZ = min(proj[0][2], proj[1][2], proj[2][2], proj[3][2])

        if type(self.scaff) == HBasicScaff:
            topP = self.scaff.t_patch.coords
            maxXT = max(topP[0][0], topP[1][0], topP[2][0], topP[3][0])
            minXT = min(topP[0][0], topP[1][0], topP[2][0], topP[3][0])
            maxYT = max(topP[0][1], topP[1][1], topP[2][1], topP[3][1])
            minYT = min(topP[0][1], topP[1][1], topP[2][1], topP[3][1])
            maxZT = max(topP[0][2], topP[1][2], topP[2][2], topP[3][2])
            minZT = min(topP[0][2], topP[1][2], topP[2][2], topP[3][2])
            # print("Checking HSCAFF")
            # print(maxXT)

            maxX = max(maxX, maxXT)
            minX = min(minX, minXT)
            maxY = max(maxY, maxYT)
            minY = min(minY, minYT)
            maxZ = max(maxZ, maxZT)
            minZ = min(minZ, minZT)
            #print(maxX)

        if (abs(np.dot(norm, XAxis)) == 1):
            bvh.append([minX + smoothbrainiqNumber, minY, minZ])
            bvh.append([minX + self.height - smoothbrainiqNumber, maxY, maxZ])
        elif (abs(np.dot(norm, YAxis)) == 1):
            print("Y Option")
            bvh.append([minX, minY + smoothbrainiqNumber, minZ])
            bvh.append([maxX, minY + self.height - smoothbrainiqNumber, maxZ])
        elif (abs(np.dot(norm, ZAxis)) == 1):
            bvh.append([minX, minY, minZ + smoothbrainiqNumber])
            bvh.append([maxX, maxY, minZ + self.height - smoothbrainiqNumber])
        else:
            raise Exception("Couldn't find fold direction when constructing BVH")

        self.bvh = np.array(bvh)

    def randomstuffIdontwannadelete(self):
         # print("OPTION==================")
        # print("Checking option against BVH")
        # print(min)
        # print(max)
        # print(self.isleft)
        # print("Height")
        # print(self.height)
        # print("num patches")
        # print(self.modification.num_hinges)

        # box = [np.array([min[0], min[1], min[2]]),
        #        np.array([min[0], min[1], max[2]]),
        #        np.array([min[0], max[1], min[2]]),
        #        np.array([min[0], max[1], max[2]]),
        #        np.array([max[0], min[1], min[2]]),
        #        np.array([max[0], min[1], max[2]]),
        #        np.array([max[0], max[1], min[2]]),
        #        np.array([max[0], max[1], max[2]])]

        # indices = [[0, 2, 4, 6],
        #            [1, 3, 5, 7],
        #            [0, 1, 4, 5],
        #            [2, 3, 6, 7],
        #            [0, 1, 2, 3],
        #            [4, 5, 6, 7]]

        # for points in indices:
        #     face = np.array([box[points[0]], box[points[1]], box[points[2]], box[points[3]]])
        #     print("COMPARING=================")
        #     print(patch.coords)
        #     print(face)
        #     if not rectangle_intersect(patch.coords, face):
        #         print("REJECTED")
        #         return False

        # We are going to cheat a little here and levrage the fact that the scaffolds only exist 
        # aligned to a particular axis
        pass 

    def checkPatchAgainstBVH(self, patch):
        print("checking patch against BVH")
        boxmin = self.bvh[0]
        boxmax = self.bvh[1]
        proj = patch.coords

        norm1 = calc_normal(proj)

        isXAxis: bool = abs(np.dot(norm1, XAxis)) == 1
        isYAxis: bool = abs(np.dot(norm1, YAxis)) == 1
        isZAxis: bool = abs(np.dot(norm1, ZAxis)) == 1

        
        maxX = max(proj[0][0], proj[1][0], proj[2][0], proj[3][0])
        minX = min(proj[0][0], proj[1][0], proj[2][0], proj[3][0])
        maxY = max(proj[0][1], proj[1][1], proj[2][1], proj[3][1])
        minY = min(proj[0][1], proj[1][1], proj[2][1], proj[3][1])
        maxZ = max(proj[0][2], proj[1][2], proj[2][2], proj[3][2])
        minZ = min(proj[0][2], proj[1][2], proj[2][2], proj[3][2])


        print("OPTION==================")
        print("Checking option against BVH")
        print(boxmin)
        print(boxmax)
        print(self.isleft)
        print("Height")
        print(self.height)
        print("num patches")
        print(self.modification.num_hinges)
        print("Coords")
        print(patch.coords)
        if isXAxis:
            if boxmin[0] >= maxX or boxmax[0] <= minX:
                return True
            else:
                if boxmax[1] <= minY or boxmin[1] >= maxY:
                    return True
                if boxmax[2] <= minZ or boxmin[2] >= maxZ:
                    return True
                
        elif isYAxis:
            if boxmin[1] >= maxY or boxmax[1] <= minY:
                return True
            else:
                if boxmax[0] <= minX or boxmin[0] >= maxX:
                    return True
                if boxmax[2] <= minZ or boxmin[2] >= maxZ:
                    return True
        elif isZAxis:
            if boxmin[2] >= maxZ or boxmax[2] <= minZ:
                return True
            else:
                if boxmax[0] <= minX or boxmin[0] >= maxX:
                    return True
                if boxmax[1] <= minY or boxmin[1] >= maxY:
                    return True
        else:
            raise Exception("Non axis aligned patch detected. Please do not do this!")
        print("REJECTED")
        return False

    def checkSelfAgainstSequenced(self, basic_scaffs: List[BasicScaff]):
        # TODO: Handle edge case
        print("Check self against sequenced? but not implemented")
        return True

    def checkSelfAgainstUnsequenced(self, basic_scaffs: List[BasicScaff]):
        print("checking self against unsequened...")
        for bs in basic_scaffs:
            if type(bs) == HBasicScaff:
                base = bs.b_patch
                fold = bs.f_patch
                top = bs.t_patch

                if top.id == self.scaff.b_patch.id:
                    continue

                if type(self.scaff) == HBasicScaff:
                    if base.id == self.scaff.t_patch.id:
                        continue

                if not self.checkPatchAgainstBVH(base) or not self.checkPatchAgainstBVH(
                        fold) or not self.checkPatchAgainstBVH(top):
                    print("Returning false at 635")
                    return False
            else:
                base = bs.b_patch
                fold = bs.f_patch

                if type(self.scaff) == HBasicScaff:
                    if base.id == self.scaff.t_patch.id:
                        continue

                if not self.checkPatchAgainstBVH(base) or not self.checkPatchAgainstBVH(fold):
                    return False
        print("ACCEPTED")
        return True

    def checkValid(self, ownedIndex: int, mid_scaffs: List[MidScaff], folded_scaff: List[bool]) -> (bool, float):
        sequenced_scaffs = []
        unsequenced_scaffs = []
        for idx in range(0, len(mid_scaffs)):
            if idx == ownedIndex:
                print("OWNED INDEX")
                continue

            for bs in mid_scaffs[idx].basic_scaffs:
                if folded_scaff[idx]:
                    print("sequenced_scaff")
                    sequenced_scaffs.append(bs)
                else:
                    print("unsequenced_scaff")
                    unsequenced_scaffs.append(bs)
        print("UNSEQUENCED LENGTH==============")
        print(len(unsequenced_scaffs))
        print("SEQUENCED LENGTH==============")
        print(len(sequenced_scaffs))
        return (
        self.checkSelfAgainstSequenced(sequenced_scaffs) and self.checkSelfAgainstUnsequenced(unsequenced_scaffs),
        self.modification.cost)


"""
BasicScaff: Parent class for TBasicScaff and HBasicScaff
"""


class BasicScaff():
    id_incr = 0

    def __init__(self):
        self.fold_options: List[FoldOption] = []

        # Says which fold options have external conflicts
        # run init conflicts to initialize
        self.no_external_conflicts: List[bool] = []
        self.id = BasicScaff.id_incr
        BasicScaff.id_incr += 1

        # To be filled by the MWISP
        self.optimal_fold_option = None
        self.conflict_id = -1

        self.aabb = self.gen_aabb()

        # To be filled by gen_fold_times
        # TODO: this logic may not even be correct to have them here.
        self.start_time = -1
        self.end_time = -1

        # offset dictated by input scaffold fold order, set by the mid level scaffold
        self.offset = -1

    def setOffset(self, offset: int):
        self.offset = offset

    def getLowestCostEsimate(self):
        low = bigChungusNumber
        for option in self.fold_options:
            if option.modification.cost < low:
                low = option.modification.cost
        return low

    def getBestOption(self):
        cost = bigChungusNumber
        base = None
        for option in self.fold_options:
            if option.modification.cost < cost:
                base = option

        return [base]

    def initConflicts(self):
        self.no_external_conflicts = [False for i in range(len(self.fold_options))]

    def clearConflicts(self):
        self.no_external_conflicts = []

    def checkValid(self, ownedIndex: int, mid_scaffs: List[MidScaff], folded_scaff: List[bool]) -> (bool, float):
        # Iterating all fold options to check if one can be foldabalized with current configuration
        min_cost = bigChungusNumber
        print("Checking valid in Basic Scaff")
        print(len(self.fold_options))
        if (len(self.fold_options) == 0):
            raise Exception("No fold options! yet checking valid...")

        for option_idx in range(0, len(self.fold_options)):
            if self.no_external_conflicts[option_idx]:
                print("Already checked")
                continue
            else:
                print("Checking external conflicts isvalid 734")
                self.no_external_conflicts[option_idx], cost = self.fold_options[option_idx].checkValid(ownedIndex,
                                                                                                        mid_scaffs,
                                                                                                        folded_scaff)
                    
                if cost < min_cost:
                    min_cost = cost

        for status in self.no_external_conflicts:

            if status:
                print("IS VALID")
                return (True, min_cost)
        return (False, min_cost)


"""
TBasicScaff: A basic scaffold of type T
"""


class TBasicScaff(BasicScaff):
    def __init__(self, b_patch, f_patch):
        self.f_patch = f_patch
        self.b_patch = b_patch

        self.rot_axis = np.cross(calc_normal(f_patch.coords), calc_normal(b_patch.coords))

    def gen_aabb(self) -> list[list[float]]:
        maxX = self.f_patch.coords[0][0]
        minX = self.f_patch.coords[0][0]
        maxY = self.f_patch.coords[0][1]
        minY = self.f_patch.coords[0][1]
        maxZ = self.f_patch.coords[0][2]
        minZ = self.f_patch.coords[0][2]

        for coord in self.f_patch.coords:
            if coord[0] > maxX:
                maxX = coord[0]
            elif coord[0] < minX:
                minX = coord[0]
            elif coord[1] > maxY:
                maxY = coord[1]
            elif coord[1] < minY:
                minY = coord[1]
            elif coord[2] > maxZ:
                maxZ = coord[2]
            elif coord[2] < minZ:
                minZ = coord[2]

        # for coord in self.f_patch.coords:
        #     if coord[0] > maxX:
        #         maxX = coord[0]
        #     elif coord[0] < minX:
        #         minX = coord[0]
        #     elif coord[1] > maxY:
        #         maxY = coord[1]
        #     elif coord[1] < minY:
        #         minY = coord[1]
        #     elif coord[2] > maxZ:
        #         maxZ = coord[2]
        #     elif coord[2] < minZ:
        #         minZ = coord[2]

        return [[minX, minY, minZ], [maxX, maxY, maxZ]]

    def gen_fold_options(self, nh, mh, ns, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        for i in range(mh, nh + 1, 2):
            for j in range(1, ns + 1):
                for h in range(1, j + 1):
                    for k in range(0, h):
                        cost = alpha * i / nh + (1 - alpha) * ((ns - (h - k)) / ns) ** 2

                        # TODO: check this later
                        # mod = Modification(i, j, k, j - k, cost)
                        mod = Modification(i, k, h, j, cost)

                        fo_left = FoldOption(True, mod, self, self.rot_axis)
                        fo_right = FoldOption(False, mod, self, self.rot_axis)

                        self.fold_options.append(fo_left)
                        self.fold_options.append(fo_right)


"""
HBasicScaff: A basic scaffold of type H
"""


class HBasicScaff(BasicScaff):
    def __init__(self, b_patch, f_patch, t_patch):
        self.f_patch = f_patch
        self.b_patch = b_patch
        self.t_patch = t_patch

        super().__init__()

        self.rot_axis: np.ndarray = np.cross(calc_normal(f_patch.coords), calc_normal(b_patch.coords))

    def gen_aabb(self) -> list[list[float]]:
        maxX = self.f_patch.coords[0][0]
        minX = self.f_patch.coords[0][0]
        maxY = self.f_patch.coords[0][1]
        minY = self.f_patch.coords[0][1]
        maxZ = self.f_patch.coords[0][2]
        minZ = self.f_patch.coords[0][2]

        # for coord in self.b_patch.coords:
        #     if coord[0] > maxX:
        #         maxX = coord[0]
        #     elif coord[0] < minX:
        #         minX = coord[0]
        #     elif coord[1] > maxY:
        #         maxY = coord[1]
        #     elif coord[1] < minY:
        #         minY = coord[1]
        #     elif coord[2] > maxZ:
        #         maxZ = coord[2]
        #     elif coord[2] < minZ:
        #         minZ = coord[2]

        for coord in self.f_patch.coords:
            if coord[0] > maxX:
                maxX = coord[0]
            elif coord[0] < minX:
                minX = coord[0]
            elif coord[1] > maxY:
                maxY = coord[1]
            elif coord[1] < minY:
                minY = coord[1]
            elif coord[2] > maxZ:
                maxZ = coord[2]
            elif coord[2] < minZ:
                minZ = coord[2]

        for coord in self.t_patch.coords:
            if coord[0] > maxX:
                maxX = coord[0]
            elif coord[0] < minX:
                minX = coord[0]
            elif coord[1] > maxY:
                maxY = coord[1]
            elif coord[1] < minY:
                minY = coord[1]
            elif coord[2] > maxZ:
                maxZ = coord[2]
            elif coord[2] < minZ:
                minZ = coord[2]

        return [[minX, minY, minZ], [maxX, maxY, maxZ]]

    def gen_fold_options(self, nh, mh, ns, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # nh: max number of hinges, let's enforce this to be an odd number for now
        # ns: max number of patch cuts

        # print("GEN FOLD OPTIONS FOR SCAFF: " + str(self.id) + "==================")

        # TODO: hard coded only doing odd number of hinges. Even numbers are too unpredictable for this purpose.
        for i in range(mh, nh + 1, 2):
            for j in range(1, ns + 1):
                for h in range(1, j + 1):
                    for k in range(0, h):
                        # print("creating modifications ------------")
                        # print("num hinges: " + str(i))
                        # print("num shrinks: " + str(j))
                        # print("start range: " + str(k))
                        # print("end range: " + str(h))
                        #
                        # print("max hinges: " + str(nh))
                        # print("max shrinks: " + str(ns))

                        cost = alpha * i / nh + (1 - alpha) * ((j - (h - k)) / j) ** 2

                        # print("cost of hinges: " + str(alpha * i / nh))
                        # print("cost of shrinks: " + str((1 - alpha) * ((j - (h - k)) / j)**2))
                        # print("cost: " + str(cost))

                        # TODO: Check if this is correct
                        # mod = Modification(i, j, k, j - k, cost)
                        mod = Modification(i, k, h, j, cost)

                        fo_left = FoldOption(True, mod, self, self.rot_axis)
                        fo_right = FoldOption(False, mod, self, self.rot_axis)

                        self.fold_options.append(fo_left)
                        self.fold_options.append(fo_right)


"""
MidScaff: a mid level folding unit that contains basic scaffolds
"""


class MidScaff:
    def __init__(self, bs, nm):
        self.basic_scaffs: List[BasicScaff] = bs
        self.node_mappings = nm
        self.conflict_graph = None  # TODO: actually the complement of the conflict graph

        # TODO: this variable will influence the fold time of the basic scaffolds
        self.start_time = 0
        self.end_time = 90

        # An offset value that is to be read by foldNode
        self.offset = -1

        self.non_conflicting_options = []

        self.best_clique = []

    def setOffset(self, offset: int):
        self.offset = offset
        for basic_scaff in self.basic_scaffs:
            basic_scaff.setOffset(offset)

    def getLowCostEstimate(self):
        cost = 0
        for scaff in self.basic_scaffs:
            cost += scaff.getLowestCostEsimate()
        return cost

    def checkValid(self, ownedIndex: int, mid_scaffs: List[MidScaff], folded_scaff: List[bool]):
        print("952 checkValid...")
        cost = 0
        for bs in self.basic_scaffs:
            valid, lowest_cost = bs.checkValid(ownedIndex, mid_scaffs, folded_scaff)
            cost += lowest_cost
            if not valid:
                print("955 checkValid: False")
                return False, bigChungusNumber
        return True, cost

    def initConflictChecks(self):
        for bs in self.basic_scaffs:
            bs.initConflicts()

    def clearConflictChecks(self):
        for bs in self.basic_scaffs:
            bs.clearConflicts()


class TMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

    def gen_fold_times(self):
        # There should only be one scaffold
        self.basic_scaffs[0].start_time = self.start_time
        self.basic_scaffs[0].end_time = self.end_time

    def build_execute_conflict_graph(self):
        clique, weight = self.basic_scaffs[0].getBestOption()
        self.best_clique = clique
        return clique, weight

    # Only one solution? Is there any case where this solution could impact another guy?
    # Probably not since they all fold at different times? Sequentially?
    def gen_conflict_graph(self):
        # Should be one huge clique
        # Assuming at this point we havea  list of basic scaffolds
        if (len(self.basic_scaffs) == 0):
            raise Exception("No basic scaffolds to generate conflict graph from!")

        # For each scaffold, get its fold options and add them as nodes to the conflict grpah
        nodes = []
        node_weights = {}

        # Sum of all the folded areas
        sum_fold_area = 0

        # Cost of the most expensive solution
        max_cost_v = -1

        for scaff in self.basic_scaffs:
            patch_area = rectangle_area(scaff.f_patch.coords)
            sum_fold_area += patch_area
            # Sum of the final area of every modification
            for option in scaff.fold_options:
                max_cost_v = max(max_cost_v, option.modification.cost)

        for scaff in self.basic_scaffs:
            for option in scaff.fold_options:
                patch_area = rectangle_area(scaff.f_patch.coords)
                lambda_i = patch_area / sum_fold_area
                cost_vj = lambda_i * option.modification.cost

                weight = max_cost_v - cost_vj + 1

                # Need to convert the weight from a double to an int by multiplying by 100
                weight = int(weight * 100)

                nodes.append(option)
                node_weights[option] = weight

        self.conflict_graph = nx.complete_graph(nodes)
        nx.set_node_attributes(self.conflict_graph, node_weights, "weight")

        # Now add the edges between each node
        # All nodes should be of the FoldOption type
        for option in self.conflict_graph.nodes:
            for other_option in self.conflict_graph.nodes:
                # if the two options are not the same and their relationship hasn't been evaluated yet
                if option != other_option and self.conflict_graph.has_edge(option, other_option) == True:
                    # Always remove. There should be no cliques in this graph.

                    self.conflict_graph.remove_edge(option, other_option)

        print("done generating conflict graph")

    def run_mwisp(self):
        # TODO: should be able to simplify
        max_clique, weight = nx.algorithms.clique.max_weight_clique(self.conflict_graph, weight="weight")

        print("RESULTING MAX CLIQUE")
        print(max_clique)

        for fold_option in max_clique:
            # TODO: clean this up messy.
            fold_option.scaff.optimal_fold_option = fold_option
            print("FOLD SOLUTION FOR: " + str(fold_option.scaff.id) + "===================")
            sol: FoldOption = fold_option
            print("start time:")
            print(fold_option.scaff.start_time)
            print("end time:")
            print(fold_option.scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(fold_option.scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

    def fold(self):
        print("TMidScaff: Implement me!")
        self.gen_conflict_graph()
        # print conflict graph's edges
        # print("conflict graph edges:")
        # for edge in self.conflict_graph.edges:
        # n1, n2 = edge
        # print("NO CONFLICT BETWEEN =======================")
        # print("SCAFF 1---------")
        # print("scaffold id: " + str(n1.scaff.id))
        # # print("num hinges: " + str(n1.modification.num_hinges))
        # # print("num shrinks: " + str(n1.modification.num_pieces))
        # # print("start range: " + str(n1.modification.range_start))
        # # print("end range: " + str(n1.modification.range_end))
        # # print("isleft: " + str(n1.isleft))
        # # # print("original vertices: ")
        # # # print(n1.scaff.f_patch.coords)
        # # print("projected region: ")
        # # print(n1.projected_region)
        # print("SCAFF 2----------")
        # print("scaffold id: " + str(n2.scaff.id))
        # # print("num hinges: " + str(n2.modification.num_hinges))
        # # print("num shrinks: " + str(n2.modification.num_pieces))
        # # print("start range: " + str(n2.modification.range_start))
        # # print("end range: " + str(n2.modification.range_end))
        # # print("isleft: " + str(n2.isleft))
        # # print("projected region: ")
        # # print(n2.projected_region)
        #
        # # print("SAME SCAFFOLD:")
        # # print(n1.scaff.id == n2.scaff.id)
        # # print("OVERLAPS:")
        # # print(rectangleOverlap(n1.projected_region, n2.projected_region))

        self.run_mwisp()

        print("Completed running fold on TMidScaff...")


class HMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

    def gen_fold_times(self, push_dir: list[float]):
        print("GENERATING FOLD TIMES....")
        # If no basic scaffolds, return with error.
        if len(self.basic_scaffs) == 0:
            raise Exception("No basic scaffolds to gen fold time for!")

        # Sort basic scaffold by the position of their top patch (This only happens for HMidScaff so we can assume there is one)
        # TODO: hard coded in the Y direction, but need to generalize this
        # Sort by the Y coordinate of the top patch, which in our case should be constant for now.
        # In-place sorting should be okay for now...
        if (abs(np.dot(push_dir, XAxis)) == 1):
            self.basic_scaffs.sort(reverse=True, key=lambda x: x.t_patch.coords[0][0])
            print("SORTED BY X")
            heightIndex = 0
        elif (abs(np.dot(push_dir, YAxis)) == 1):
            print("SORTED BY Y")
            self.basic_scaffs.sort(reverse=True, key=lambda x: x.t_patch.coords[0][1])
            heightIndex = 1
        elif (abs(np.dot(push_dir, ZAxis)) == 1):
            self.basic_scaffs.sort(reverse=True, key=lambda x: x.t_patch.coords[0][2])
            print("SORTED BY Z")
            heightIndex = 2
        else:
            raise Exception("Push direction is not a cardinal direction!")

        # self.basic_scaffs.sort(reverse=True, key=lambda x: x.t_patch.coords[0][1])

        highest_patch = self.basic_scaffs[0].t_patch
        lowest_patch = self.basic_scaffs[-1].b_patch
        print("highest patch coords:")
        print(highest_patch.coords)
        h0 = highest_patch.coords[0][heightIndex]
        b0 = lowest_patch.coords[0][heightIndex]

        print("h0: " + str(h0))

        for basic_scaff in self.basic_scaffs:
            # TODO: hard coded for the Y direction, but need to generalize
            h = basic_scaff.t_patch.coords[0][heightIndex]
            b = basic_scaff.b_patch.coords[0][heightIndex]

            print("h: " + str(h))
            print("b: " + str(b))

            # print("basic scaff:")
            # print(basic_scaff)

            print("h0 - h: " + str(h0 - h))
            print("h0 - b: " + str(h0 - b))

            basic_scaff.start_time = abs(h0 - h) * self.end_time / abs(h0 - b0)
            print("start time: " + str(basic_scaff.start_time))
            basic_scaff.end_time = abs(h0 - b) * self.end_time / abs(h0 - b0)
            print("end time: " + str(basic_scaff.end_time))

        print("Finished generating fold times for HMidScaff")

        self.added_tracker = []
        self.fold_Order = []

    def gen_conflict_graph(self):
        print("generating conflict graph...")

        # Assuming at this point we havea  list of basic scaffolds
        if (len(self.basic_scaffs) == 0):
            raise Exception("No basic scaffolds to generate conflict graph from!")

        # For each scaffold, get its fold options and add them as nodes to the conflict grpah
        nodes = []
        node_weights = {}

        # Sum of all the folded areas
        sum_fold_area = 0

        # Cost of the most expensive solution
        max_cost_v = -1

        for scaff in self.basic_scaffs:
            patch_area = rectangle_area(scaff.f_patch.coords)
            sum_fold_area += patch_area
            # Sum of the final area of every modification
            for option in scaff.fold_options:
                max_cost_v = max(max_cost_v, option.modification.cost)

        for (scaff, idx) in zip(self.basic_scaffs, range(0, len(self.basic_scaffs))):
            for option in scaff.fold_options:
                # Ensuring that the option has no external conflicts
                if scaff.no_external_conflicts[idx]:
                    patch_area = rectangle_area(scaff.f_patch.coords)
                    lambda_i = patch_area / sum_fold_area
                    cost_vj = lambda_i * option.modification.cost

                    weight = max_cost_v - cost_vj + 1

                    # print weight and fold option elements
                    # print("COST of the NODE: " + str(scaff.id) + " is " + str(option.modification.cost))
                    # print("max_cost_v: " + str(max_cost_v))
                    # print("weight: " + str(weight))
                    # print("num hinges: ")
                    # print(option.modification.num_hinges)
                    # print("num shrinks: ")
                    # print(option.modification.num_pieces)
                    # print("start range:")
                    # print(option.modification.range_start)
                    # print("end range:")
                    # print(option.modification.range_end)

                    # print("original vertices: ")
                    # print(scaff.f_patch.coords)
                    # print("Projected region of solution: ")
                    # print(sol.projected_region)

                    # Need to convert the weight from a double to an int by multiplying by 100
                    weight = int(weight * 100)

                    nodes.append(option)
                    node_weights[option] = weight

        self.conflict_graph = nx.complete_graph(nodes)
        nx.set_node_attributes(self.conflict_graph, node_weights, "weight")

        # Now add the edges between each node
        # All nodes should be of the FoldOption type
        for option in self.conflict_graph.nodes:
            for other_option in self.conflict_graph.nodes:
                # if the two options are not the same and their relationship hasn't been evaluated yet
                if option != other_option and self.conflict_graph.has_edge(option, other_option) == True:
                    # Check if the two nodes conflict
                    if option.conflicts_with(other_option):
                        # if they conflict, remove this edge
                        self.conflict_graph.remove_edge(option, other_option)

                    # # TODO: debug test for now, to see which nodes will pop out
                    # if (option.scaff.id == other_option.scaff.id):
                    #     self.conflict_graph.remove_edge(option, other_option)

        print("done generating conflict graph")

    def run_mwisp(self):
        print("running MWISP")

        # Run MWISP on the conflict graph
        # This will return a list of nodes (FoldOptions) that are in the maximum weight independent set
        # TODO: this might be really slow.
        max_clique, weight = nx.algorithms.clique.max_weight_clique(self.conflict_graph, weight="weight")

        print("RESULTING MAX CLIQU")
        print(max_clique)

        if len(max_clique) != len(self.basic_scaffs):
            raise Exception("Max clique is missing some options")

        for fold_option in max_clique:
            # TODO: clean this up messy.
            fold_option.scaff.optimal_fold_option = fold_option
            print("FOLD SOLUTION FOR: " + str(fold_option.scaff.id) + "===================")
            sol: FoldOption = fold_option
            print("start time:")
            print(fold_option.scaff.start_time)
            print("end time:")
            print(fold_option.scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(fold_option.scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

        return max_clique, weight

    def build_execute_conflict_graph(self):
        self.gen_conflict_graph()
        clique, weight = self.run_mwisp()
        self.best_clique = clique
        return clique, weight

    def fold(self):
        self.gen_conflict_graph()

        # print conflict graph's edges
        # print("conflict graph edges:")
        # for edge in self.conflict_graph.edges:
        # n1, n2 = edge
        # print("NO CONFLICT BETWEEN =======================")
        # print("SCAFF 1---------")
        # print("scaffold id: " + str(n1.scaff.id))
        # # print("num hinges: " + str(n1.modification.num_hinges))
        # # print("num shrinks: " + str(n1.modification.num_pieces))
        # # print("start range: " + str(n1.modification.range_start))
        # # print("end range: " + str(n1.modification.range_end))
        # # print("isleft: " + str(n1.isleft))
        # # # print("original vertices: ")
        # # # print(n1.scaff.f_patch.coords)
        # # print("projected region: ")
        # # print(n1.projected_region)
        # print("SCAFF 2----------")
        # print("scaffold id: " + str(n2.scaff.id))
        # # print("num hinges: " + str(n2.modification.num_hinges))
        # # print("num shrinks: " + str(n2.modification.num_pieces))
        # # print("start range: " + str(n2.modification.range_start))
        # # print("end range: " + str(n2.modification.range_end))
        # # print("isleft: " + str(n2.isleft))
        # # print("projected region: ")
        # # print(n2.projected_region)
        #
        # # print("SAME SCAFFOLD:")
        # # print(n1.scaff.id == n2.scaff.id)
        # # print("OVERLAPS:")
        # # print(rectangleOverlap(n1.projected_region, n2.projected_region))

        self.run_mwisp()

        print("Completed running fold on HMidScaff...")


"""
InputScaff: The full input scaff
"""


class InputScaff:
    id_incr = 0

    def __init__(self, node_list, edge_list, push_dir, max_hinges, min_hinges, num_shrinks, alpha):
        self.hinge_graph = None  # Gets created gen_hinge_graph
        self.mid_scaffs: List[MidScaff] = []

        self.mid_scaffs_ordered: List[MidScaff] = []
        self.folded_scaff: List[bool] = []

        # patch list
        self.node_list = node_list
        # refers to indices in patch list
        self.edge_list = edge_list
        # axis vec3
        self.push_dir: np.ndarray = push_dir

        # debug purposes for ease of our test algorithm
        # for now we manually define basic scaffolds
        self.basic_scaffs = []

        # This is for mapping for basic scaffs ids (ints) to node ids (list of ints)
        self.basic_mappings = {}

        self.max_hinges = max_hinges
        self.min_hinges = min_hinges
        self.num_shrinks = num_shrinks

        self.alpha = alpha

    # TODO: FOR UNIT TESTING PURPOSES ONLY DO NOT USE IN PRODUCTION
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
                    if (self.push_dir == XAxis).all():
                        if (base1.coords[0][0] > base2.coords[0][0]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    elif (self.push_dir == YAxis).all():
                        if (base1.coords[0][1] > base2.coords[0][1]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    elif (self.push_dir == ZAxis).all():
                        if (base1.coords[0][2] > base2.coords[0][2]):
                            base_hi = base1
                            base_lo = base2
                        else:
                            base_hi = base2
                            base_lo = base1
                    else:
                        raise Exception("Invalid push direction")
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

    # TODO: ONLY USED FOR UNIT TESTING
    # def gen_scaffs(self):
    #     # TODO: Di
    #
    #     # generates hinge graph
    #     self.gen_hinge_graph()
    #
    #     # generates basic scaffolds
    #     # self.gen_basic_scaffs()
    #
    #     # generates mid-level scaffolds
    #     self.gen_mid_scaffs()
    #
    #     # generates fold order. Fold order will be stored in self.mid_scaff_ordered
    #     # Midscaffs will also store the best clique  in Midscaff.best_clique
    #     # self.order_folds()

    def gen_hinge_graph(self):
        print("gen_hinge_graph...")
        for patch in self.node_list:
            ang = abs(np.dot(normalize(self.push_dir), normalize(patch.normal)))
            # print(normalize(patch.normal))
            if ang < .1:
                patch.patch_type = PatchType.Fold
            else:
                patch.patch_type = PatchType.Base

        self.hinge_graph = nx.Graph()
        for patch in self.node_list:
            self.hinge_graph.add_node(patch.id)

        for edge in self.edge_list:
            self.hinge_graph.add_edge(edge[0], edge[1])

    # Here for debugging purposes
    # def gen_basic_scaffs(self):
    #     print("gen basic scaffs")
    #     for patch in self.node_list:
    #         if patch.patch_type == PatchType.Fold:
    #             id = patch.id
    #             neighbors = list(self.hinge_graph.neighbors(id))
    #             if len(neighbors) == 2:
    #                 base0 = self.node_list[neighbors[0]]
    #                 base1 = self.node_list[neighbors[1]]
    #                 fold0 = self.node_list[id]
    #                 self.basic_scaffs.append(HBasicScaff(base0, fold0, base1))
    #                 self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id,
    #                                                                  self.node_list[neighbors[1]].id]
    #                 # print(self.basic_scaffs[-1].id)
    #             elif len(neighbors) == 1:
    #                 base0 = self.node_list[neighbors[0]]
    #                 fold0 = self.node_list[id]
    #                 self.basic_scaffs.append(TBasicScaff(base0, fold0))
    #                 self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id]
    #                 # print(self.basic_scaffs[-1].id)
    #             else:
    #                 print("wtf")
    #
    #     for scaff in self.basic_scaffs:
    #         # TODO: Remove some of these from debugging
    #         scaff.start_time = 0
    #         scaff.end_time = 90
    #         scaff.gen_fold_options(self.num_shrinks, self.max_hinges, .5)
    #
    #     # print("end gen basic scaffs")

    # Basic scaffold objects already created by the foldNode
    def set_basic_mappings(self):
        print("gen basic scaffs")
        if (len(self.basic_scaffs) < 1):
            raise Exception("No basic scaffolds!")

        # At this point there should be at least one basic scaffold in here
        for basic_scaff in self.basic_scaffs:
            if (type(basic_scaff) is HBasicScaff):
                scaffid = basic_scaff.id
                patchid = basic_scaff.f_patch.id
                # Produce basic mappings in the same way I produced them foldNode.
                print("scaffid: " + str(scaffid))
                print("patch ids: " + str(basic_scaff.b_patch.id) + ", " + str(basic_scaff.t_patch.id))
                self.basic_mappings[scaffid] = [patchid, basic_scaff.b_patch.id, basic_scaff.t_patch.id]
            elif (type(basic_scaff) is TBasicScaff):
                scaffid = basic_scaff.id
                patchid = basic_scaff.f_patch.id
                self.basic_mappings[scaffid] = [patchid, basic_scaff.b_patch.id]
            else:
                raise Exception("Unknown basic scaff type!")

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
        # If there are no basic scaffolds, return with an error
        if len(self.basic_scaffs) == 0:
            raise Exception("No basic scaffolds to generate mid level scaffolds from")

        if len(self.basic_mappings) == 0:
            raise Exception("No basic scaffold mappings")

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
            # print("analyzing cycle...")
            # print(cycle)

            basic_scaff_id_list = []
            patch_id_list = []

            # A basic scaffold is part of a mid level scaffolds if it's part of a cycle
            for scaff_id, patch_ids in self.basic_mappings.items():
                # print("scaff_id: " + str(scaff_id))
                # print("patch_ids: " + str(patch_ids))

                reject = False
                patch_id_temp = []
                for id in patch_ids:
                    # print("id: " + str(id))
                    if id not in cycle:
                        # print("rejecting id...")
                        reject = True
                        break
                    else:
                        patch_id_temp.append(id)
                if not reject:
                    # print("adding scaff_id: " + str(scaff_id))
                    basic_scaff_id_list.append(scaff_id)
                    tracker[scaff_id] = True
                    patch_id_list = patch_id_list + patch_id_temp
            pruned_id_list = [*set(patch_id_list)]

            basic_scaff_list = []
            for scaff_id in basic_scaff_id_list:
                basic_scaff_list.append(self.basic_scaffs[scaff_id])

            if (len(basic_scaff_list) == 0):
                raise Exception("Found no basic scaffolds to generate Mid level scaffolds")
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

    def gen_fold_options(self):
        # TODO: For now, only foldabilize the first mid level scaffold.
        # TODO: Eventually I will need to do greedy one step lookahead.

        # If no scaffolds, return with an error
        if len(self.basic_scaffs) == 0:
            raise Exception("No basic scaffolds to fold")

        # First, generate time zones for basic scaffolds
        for mid_scaff in self.mid_scaffs:
            mid_scaff.gen_fold_times(self.push_dir)

        # First, generate basic scaffold solutions
        for scaff in self.basic_scaffs:
            # TODO: hard code alpha to be 0.5
            scaff.gen_fold_options(self.max_hinges, self.min_hinges, self.num_shrinks, self.alpha)

        # self.mid_scaffs[0].fold()

    def initConflictChecks(self):
        for ms in self.mid_scaffs:
            ms.initConflictChecks()

    def clearConflictChecks(self):
        for ms in self.mid_scaffs:
            ms.clearConflictChecks()

    def checkValid(self, index):
        # TODO: IMPLEMENT
        print("1643 Check Valid")
        return self.mid_scaffs[index].checkValid(index, self.mid_scaffs, self.folded_scaff)

    def pickNextScaff(self):
        c1 = bigChungusNumber
        c2 = bigChungusNumber
        best_Scaff = None
        bestId = -1

        for (scaff1, id1) in zip(self.mid_scaffs, range(0, len(self.mid_scaffs))):
            # check valid sees if the mid scaff can be folded
            print("In for loop checking: " + str(id1))

            # If folded_scaff doesn't contain id1
            if not self.folded_scaff[id1]:
                valid, first_cost = self.checkValid(id1)
                if not valid:
                    print("Solution was not valid!")
                    continue

                weight1 = first_cost
                for (scaff2, id2) in zip(self.mid_scaffs, range(0, len(self.mid_scaffs))):
                    weight2 = self.mid_scaffs[id2].getLowCostEstimate()
                    if id1 != id2 and c1 + c2 > weight1 + weight2:
                        print("Setting bestId " + str(id1))
                        c1 = weight1
                        c2 = weight2
                        best_Scaff = scaff1
                        bestId = id1
                    # Case where we have one left
                    elif id1 == id2 and len(self.mid_scaffs_ordered) == len(self.mid_scaffs) - 1:
                        self.mid_scaffs_ordered.append(scaff1)
                        self.folded_scaff[id1] = True
                        return

        print("Adding mid level scaff: " + str(bestId))

        if best_Scaff == None:
            remaining_list = []
        
        self.mid_scaffs_ordered.append(best_Scaff)
        self.folded_scaff[bestId] = True

    def order_folds(self):
        size = len(self.mid_scaffs)
        print("number of mid level scaffs:")
        print(size)

        self.folded_scaff = [False for i in range(size)]
        self.initConflictChecks()

        print("number of ordered mid level scaffolds")
        print(len(self.mid_scaffs_ordered))

        offset = 0
        while len(self.mid_scaffs_ordered) < len(self.mid_scaffs):
            self.pickNextScaff()
            print("Num Ordered:")
            print(len(self.mid_scaffs_ordered))
            print(self.mid_scaffs_ordered)
            print(self.mid_scaffs_ordered[0])
            print(self.mid_scaffs_ordered[-1])
            self.mid_scaffs_ordered[-1].build_execute_conflict_graph()
            self.mid_scaffs_ordered[-1].setOffset(offset * 90)
            offset += 1

        self.clearConflictChecks()
        self.folded_scaff = [False for i in range(size)]


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

    tscaff = HBasicScaff(base, foldable, top)
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

    input = InputScaff([b1, b2, b3, f1, f2, f3], [[0, 3], [0, 5], [1, 3], [1, 4], [2, 4], [2, 5]], normalize([0, 1, 0]),
                       2, 1)

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
    coords6 = np.array([(3.25, 0, 5.5), (3.25, 0, 4.5), (3.25, 2, 4.5), (3.25, 2, 5.5)])  # right fold # 5

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
    edges = [[3, 2], [1, 3], [2, 4], [4, 0], [5, 0], [1, 5]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 5, 2)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)


# test_conflict_graph()

def test_conflict_graph_2midScaffs():
    coords1 = np.array([(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)])  # top base patch # 0
    coords2 = np.array([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)])  # bottom base patch # 1
    coords3 = np.array([(0.25, 0, 1), (0.25, 0, 0), (0.25, 1, 0), (0.25, 1, 1)])  # middle base patch # 2
    coords4 = np.array([(0.75, 0, 1), (0.75, 0, 0), (0.25, 1, 0), (0.75, 1, 1)])  # left bot fold patch # 3

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)
    f2 = Patch(coords4)

    b1.id = 0
    b2.id = 1

    f1.id = 2
    f2.id = 3

    nodes = [b1, b2, f1, f2]
    edges = [[2, 0], [1, 2], [3, 0], [1, 3]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 5, 2)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)


# test_conflict_graph_2midScaffs()

def test_conflict_graph_inconsistentFoldPatch():
    coords1 = np.array([(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)])  # top base patch # 0
    coords2 = np.array([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)])  # bottom base patch # 1
    coords3 = np.array([(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5)])  # middle base patch # 2

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)

    b1.id = 0
    b2.id = 1

    f1.id = 2

    nodes = [b1, b2, f1]
    edges = [[2, 0], [1, 2]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 5, 2)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)


# test_conflict_graph_inconsistentFoldPatch()

def test_cube_shape():
    coords1 = np.array([(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)])  # bottom base patch # 0
    coords2 = np.array([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)])  # top base patch # 1
    coords3 = np.array([(0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)])  # fold patch 1 # 2
    coords4 = np.array([(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)])  # fold patch 2 # 3
    coords5 = np.array([(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)])  # fold patch 3 # 4

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

    nodes = [b1, b2, f1, f2, f3]
    edges = [[2, 0], [3, 0], [4, 0], [1, 2], [1, 3], [1, 4]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 5, 1, 3, 0.5)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    # Generate solutions
    input.gen_fold_options()

    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)


# test_cube_shape()


def test_input_scaff():
    coords1 = np.array([(0, 2, 0), (1, 2, 0), (1, 2, 1), (0, 2, 1)])  # base 0
    coords2 = np.array([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)])  # base 1
    coords3 = np.array([(0, 0, 0), (1, 0, 0), (1,0, 1), (0, 0, 1)])  # base 2
    # coords4 = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])  # base 3

    coords5 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1)])  # fold 0
    coords6 = np.array([(0.5, 1, 1), (0.5, 1, 0), (0.5, 2, 0), (0.5, 2, 1)])  # fold 1
    # coords7 = np.array([(0.5, 2, 1), (0.5, 2, 0), (0.5, 3, 0), (0.5, 3, 1)])  # fold 2

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)
    # b4 = Patch(coords4)

    f1 = Patch(coords5)
    f2 = Patch(coords6)
    # f3 = Patch(coords7)

    b1.id = 0
    b2.id = 1
    b3.id = 2
    # b4.id = 3

    # f1.id = 4
    f1.id = 3

    # f2.id = 5
    f2.id = 4
    # f3.id = 6

    # nodes = [b1, b2, b3, b4, f1, f2, f3]
    nodes = [b1, b2, b3, f1, f2]

    # edges = [[4, 0], [5, 1], [6, 2], [3, 6], [2, 5], [1, 4]]
    edges = [[3, 0], [3, 1], [4, 1], [2, 4]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 1, 1, 2, 0.5)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    input.gen_fold_options()

    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

    for mid_scaff in input.mid_scaffs_ordered:
        print("MID SCAFF ID")
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD PATCH COORDS")
            print(basic_scaff.f_patch.coords)
# test_input_scaff()

def test_side_by_side_input():
    coords1 = np.array([(0, 0, 0), (2, 0, 0), (2, 0, 1), (0, 0, 1)])  # base 0
    coords2 = np.array([(0, 1, 0), (0.8, 1, 0), (0.8, 1, 1), (0, 1, 1)])  # base 1
    coords3 = np.array([(1.2, 1, 0), (2, 1, 0), (2, 1, 1), (1.2, 1, 1)])  # base 2

    coords4 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, .5, 0), (0.5, .5, 1)])  # fold 0
    coords5 = np.array([(1.5, 0, 1), (1.5, 0, 0), (1.5, .5, 0), (1.5, .5, 1)])  # fold 1

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)

    f1 = Patch(coords4)
    f2 = Patch(coords5)

    b1.id = 0
    b2.id = 1
    b3.id = 2

    f1.id = 3
    f2.id = 4

    nodes = [b1, b2, b3, f1, f2]

    edges = [[0, 3], [3, 1], [0, 4], [4, 2]]

    push_dir = YAxis

    # nh, ns
    input = InputScaff(nodes, edges, push_dir, 1, 1)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

# test_side_by_side_input()

def t_construct_bvh():
    coords1 = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]])
    coords2 = np.array([[0, 2, 1], [4, 2, 1], [4, 2, 0], [0, 2, 0]])
    coords3 = np.array([[0.5, 0, 1], [0.5, 0, 0], [0.5, 10, 0], [0.5, 10, 1]])

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)

    b1.id = 0
    b2.id = 1
    f1.id = 2

    nodes = [b1, b2, f1]
    edges = [[0, 2], [1, 2]]
    push_dir = YAxis
    input = InputScaff(nodes, edges, push_dir, 3, 2)
    input.gen_hinge_graph()
    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")

    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)

    input.basic_scaffs[0].start_time = 0
    input.basic_scaffs[0].end_time = 90

    # Should be the only one
    # nh, ns
    input.basic_scaffs[0].gen_fold_options(3, 1, 0.5)

    # print every fold option's bvh
    for basic_scaff in input.basic_scaffs:
        for fold_option in basic_scaff.fold_options:
            
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(fold_option.scaff.end_time)
            print("num hinges: ")
            print(fold_option.modification.num_hinges)
            print("num shrinks: ")
            print(fold_option.modification.num_pieces)
            print("range start: ")
            print(fold_option.modification.range_start)
            print("range end: ")
            print(fold_option.modification.range_end)
            print("isleft:")
            print(fold_option.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(fold_option.projected_region)

            print("BVH")
            print(fold_option.bvh)
            print("num options")
            print(len(basic_scaff.fold_options))

#t_construct_bvh()

def patch_intersect_check():
    coords1 = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]])
    coords2 = np.array([[0, 2, 1], [1, 2, 1], [1, 2, 0], [0, 2, 0]])
    coords3 = np.array([[0.5, 0, 1], [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1]])

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)

    b1.id = 0
    b2.id = 1
    f1.id = 2

    nodes = [b1, b2, f1]
    edges = [[0, 2], [1, 2]]
    push_dir = YAxis
    input = InputScaff(nodes, edges, push_dir, 3, 2)
    input.gen_hinge_graph()
    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")

    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)

    input.basic_scaffs[0].start_time = 0
    input.basic_scaffs[0].end_time = 90

    # Should be the only one
    # nh, ns
    input.basic_scaffs[0].gen_fold_options(3, 1, 0.5)

    # print every fold option's bvh
    for basic_scaff in input.basic_scaffs:
        for fold_option in basic_scaff.fold_options:
            
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(fold_option.scaff.end_time)
            print("num hinges: ")
            print(fold_option.modification.num_hinges)
            print("num shrinks: ")
            print(fold_option.modification.num_pieces)
            print("range start: ")
            print(fold_option.modification.range_start)
            print("range end: ")
            print(fold_option.modification.range_end)
            print("isleft:")
            print(fold_option.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(fold_option.projected_region)

            print("BVH")
            print(fold_option.bvh)
            print("num options")
            print(len(basic_scaff.fold_options))

def test_rectanlge_overlap():
    print("TEST RECTANGLE OVERLAP")
    coords1 = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]])
    coords2 = np.array([[0, 2, 1], [1, 2, 1], [1, 2, 0], [0, 2, 0]])
    coords3 = np.array([[0.5, 0, 1], [0.5, 0, 0], [0.5, 2, 0], [0.5, 2, 1]])

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    f1 = Patch(coords3)

    b1.id = 0
    b2.id = 1
    f1.id = 2

    nodes = [b1, b2, f1]
    edges = [[0, 2], [1, 2]]
    push_dir = YAxis
    input = InputScaff(nodes, edges, push_dir, 3, 2)
    input.gen_hinge_graph()
    for l in range(0, len(input.node_list)):
        print(input.node_list[l].id)
        print(input.node_list[l].patch_type)
        print(list(input.hinge_graph.neighbors(l)))
        print("------------")

    print(input.hinge_graph)
    input.gen_basic_scaffs()
    print(input.basic_scaffs)

    input.basic_scaffs[0].start_time = 0
    input.basic_scaffs[0].end_time = 90

    # Should be the only one
    # nh, ns
    input.basic_scaffs[0].gen_fold_options(1, 1, 0.5)

    input.basic_scaffs[0].fold_options[0].bvh = np.array([[0,0,0],[1,1,1]])
    option = input.basic_scaffs[0].fold_options[0]
    print("COMMENCE BOX TESTS")
    print(option.bvh)

    patch1 = Patch(np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]]))
    patch2 = Patch(np.array([[0, .5, 1], [1, .5, 1], [1, .5, 0], [0, .5, 0]]))
    patch3 = Patch(np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]]))
    patch4 = Patch(np.array([[1, .5, 1], [2, .5, 1], [2, .5, 0], [1, .5, 0]]))

    print("Should not intersect")
    print(option.checkPatchAgainstBVH(patch1))

    print("Should intersect")
    print(option.checkPatchAgainstBVH(patch2))

    print("Should not intersect")
    print(option.checkPatchAgainstBVH(patch3))

    print("Should not intersect")
    print(option.checkPatchAgainstBVH(patch4))

#test_rectanlge_overlap()

def test_side_by_side_one_high_input():
    coords1 = np.array([(0, 0, 0), (2, 0, 0), (2, 0, 1), (0, 0, 1)])  # base 0
    coords2 = np.array([(0, 1, 0), (0.8, 1, 0), (0.8, 1, 1), (0, 1, 1)])  # base 1
    coords3 = np.array([(1.2, 2, 0), (2, 2, 0), (2, 2, 1), (1.2, 2, 1)])  # base 2

    coords4 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1)])  # fold 0
    coords5 = np.array([(1.5, 0, 1), (1.5, 0, 0), (1.5, 2, 0), (1.5, 2, 1)])  # fold 1

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)

    f1 = Patch(coords4)
    f2 = Patch(coords5)

    b1.id = 0
    b2.id = 1
    b3.id = 2

    f1.id = 3
    f2.id = 4

    nodes = [b1, b2, b3, f1, f2]

    edges = [[0, 3], [3, 1], [0, 4], [4, 2]]

    push_dir = YAxis

    # nh, ns
    input = InputScaff(nodes, edges, push_dir, 1, 1, 1, 0.5)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    input.gen_fold_options()
    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

    for mid_scaff in input.mid_scaffs_ordered:
        print("MID SCAFF ID")
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD PATCH COORDS")
            print(basic_scaff.f_patch.coords)

# test_side_by_side_one_high_input()

def test_side_by_side_optimal_sequence_input():
    coords1 = np.array([(0, 0, 0), (2, 0, 0), (2, 0, 1), (0, 0, 1)])  # base 0
    coords2 = np.array([(0, 2, 0), (2, 2, 0), (2, 2, 1), (0, 2, 1)])  # base 1
    coords3 = np.array([(1.2, 1, 0), (2, 1, 0), (2, 1, 1), (1.2, 1, 1)])  # base 2

    coords4 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, 2, 0), (0.5, 2, 1)])  # fold 0
    coords5 = np.array([(1.5, 0, 1), (1.5, 0, 0), (1.5, 1, 0), (1.5, 1, 1)])  # fold 1

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)

    f1 = Patch(coords4)
    f2 = Patch(coords5)

    b1.id = 0
    b2.id = 1
    b3.id = 2

    f1.id = 3
    f2.id = 4

    nodes = [b1, b2, b3, f1, f2]

    edges = [[0, 3], [3, 1], [0, 4], [4, 2]]

    push_dir = YAxis

    # nh, ns
    input = InputScaff(nodes, edges, push_dir, 1, 1, 1, 0.5)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    input.gen_fold_options()

    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    for mid_scaff in input.mid_scaffs:
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

    for mid_scaff in input.mid_scaffs_ordered:
        print("MID SCAFF ID")
        for basic_scaff in mid_scaff.basic_scaffs:
            print("FOLD PATCH COORDS")
            print(basic_scaff.f_patch.coords)

# test_side_by_side_optimal_sequence_input()

def test_two_bottom_one_top():
    coords1 = np.array([(0, 2, 0), (1, 2, 0), (1, 2, 1), (0, 2, 1)])  # base 0
    coords2 = np.array([(0, 1, 0), (2, 1, 0), (2, 1, 1), (0, 1, 1)])  # base 1
    coords3 = np.array([(0, 0, 0), (2, 0, 0), (2, 0, 1), (0, 0, 1)])  # base 2

    coords5 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1)])  # fold 0
    coords6 = np.array([(0.5, 1, 1), (0.5, 1, 0), (0.5, 2, 0), (0.5, 2, 1)])  # fold 1
    coords7 = np.array([(1.5, 0, 1), (1.5, 0, 0), (1.5, 1, 0), (1.5, 1, 1)])  # fold 1

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)
    # b4 = Patch(coords4)

    f1 = Patch(coords5)
    f2 = Patch(coords6)
    f3 = Patch(coords7)

    b1.id = 0
    b2.id = 1
    b3.id = 2

    f1.id = 3
    f2.id = 4
    f3.id = 5

    nodes = [b1, b2, b3, f1, f2, f3]

    edges = [[0,4], [4,1], [1,3],[3,2],[5,2],[1,5]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 1, 1, 2, 0.5)

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
        for basic_scaff in mid_scaff.basic_scaffs:
            print("SCAFF =================== ")
            print("base high: " + str(basic_scaff.t_patch.id))
            print("foldable: " + str(basic_scaff.f_patch.id))
            print("base low: " + str(basic_scaff.b_patch.id))

    input.gen_fold_options()

    input.order_folds()

    # Generate solutions
    # input.fold()

    # Print the generated solutions
    # for mid_scaff in input.mid_scaffs:
    #     for basic_scaff in mid_scaff.basic_scaffs:
    #         print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
    #         sol: FoldOption = basic_scaff.optimal_fold_option
    #         print("start time:")
    #         print(basic_scaff.start_time)
    #         print("end time:")
    #         print(basic_scaff.end_time)
    #         print("num hinges: ")
    #         print(sol.modification.num_hinges)
    #         print("num shrinks: ")
    #         print(sol.modification.num_pieces)
    #         print("range start: ")
    #         print(sol.modification.range_start)
    #         print("range end: ")
    #         print(sol.modification.range_end)
    #         print("isleft:")
    #         print(sol.isleft)
    #         print("original vertices: ")
    #         print(basic_scaff.f_patch.coords)
    #         print("Projected region of solution: ")
    #         print(sol.projected_region)

    indexTime = 0
    for mid_scaff in input.mid_scaffs_ordered:
        print("MID SCAFF ID")
        for basic_scaff in mid_scaff.basic_scaffs:
            # print("FOLD PATCH COORDS")
            # print(basic_scaff.f_patch.coords)
            # offset = indexTime * 90

            print("FOLD SOLUTION FOR: " + str(basic_scaff.id) + "===================")
            sol: FoldOption = basic_scaff.optimal_fold_option
            print("start time:")
            print(basic_scaff.offset + basic_scaff.start_time)
            print("end time:")
            print(basic_scaff.offset + basic_scaff.end_time)
            print("num hinges: ")
            print(sol.modification.num_hinges)
            print("num shrinks: ")
            print(sol.modification.num_pieces)
            print("range start: ")
            print(sol.modification.range_start)
            print("range end: ")
            print(sol.modification.range_end)
            print("isleft:")
            print(sol.isleft)
            print("original vertices: ")
            print(basic_scaff.f_patch.coords)
            print("Projected region of solution: ")
            print(sol.projected_region)

        indexTime += 1

test_two_bottom_one_top()