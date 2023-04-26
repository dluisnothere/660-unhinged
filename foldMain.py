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


def rectangle_overlap(rect1: np.ndarray(np.ndarray(float)), rect2: np.ndarray(np.ndarray(float))) -> bool:
    # If rectangles not in the same plane or don't have the same/opposite normals, then don't overlap
    norm1 = calc_normal(rect1)
    norm2 = calc_normal(rect2)

    if (abs(np.dot(norm1, norm2)) != 1):
        return False

    isXAxis: bool = abs(np.dot(norm1, XAxis)) == 1
    isYAxis: bool = abs(np.dot(norm1, YAxis)) == 1
    isZAxis: bool = abs(np.dot(norm1, ZAxis)) == 1

    if not isXAxis and not isZAxis and not isYAxis:
        raise Exception("Rectangles are not axis aligned...?")
    else:
        # If rectangles not coplanar
        if (isXAxis):
            rect1_minY = min(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
            rect1_maxY = max(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
            rect1_minZ = min(rect1[0][2], rect1[1][2], rect1[2][2], rect1[3][2])
            rect1_maxZ = max(rect1[0][2], rect1[1][2], rect1[2][2], rect1[3][2])

            # if any vertex lies within the rectangle above, then there is some degree of overlap
            for vertex in rect2:
                if rect1_minY <= vertex[1] <= rect1_maxY and rect1_minZ <= vertex[2] <= rect1_maxZ:
                    return True

            rect2_minY = min(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])
            rect2_maxY = max(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])
            rect2_minZ = min(rect2[0][2], rect2[1][2], rect2[2][2], rect2[3][2])
            rect2_maxZ = max(rect2[0][2], rect2[1][2], rect2[2][2], rect2[3][2])

            for vertex in rect1:
                if rect2_minY <= vertex[1] <= rect2_maxY and rect2_minZ <= vertex[2] <= rect2_maxZ:
                    return True

            return False

        elif (isYAxis):
            rect1_minX = min(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
            rect1_maxX = max(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
            rect1_minZ = min(rect1[0][2], rect1[1][2], rect1[2][2], rect1[3][2])
            rect1_maxZ = max(rect1[0][2], rect1[1][2], rect1[2][2], rect1[3][2])

            # if any vertex lies within the rectangle above, then there is some degree of overlap
            for vertex in rect2:
                if rect1_minX <= vertex[0] <= rect1_maxX and rect1_minZ <= vertex[2] <= rect1_maxZ:
                    return True

            rect2_minX = min(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
            rect2_maxX = max(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
            rect2_minZ = min(rect2[0][2], rect2[1][2], rect2[2][2], rect2[3][2])
            rect2_maxZ = max(rect2[0][2], rect2[1][2], rect2[2][2], rect2[3][2])

            for vertex in rect1:
                if rect2_minX <= vertex[0] <= rect2_maxX and rect2_minZ <= vertex[2] <= rect2_maxZ:
                    return True

            return False

        elif (isZAxis):
            rect1_minX = min(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
            rect1_maxX = max(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
            rect1_minY = min(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
            rect1_maxY = max(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])

            # if any vertex lies within the rectangle above, then there is some degree of overlap
            for vertex in rect2:
                if rect1_minX <= vertex[0] <= rect1_maxX and rect1_minY <= vertex[1] <= rect1_maxY:
                    return True

            rect2_minX = min(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
            rect2_maxX = max(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
            rect2_minY = min(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])
            rect2_maxY = max(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])

            for vertex in rect1:
                if rect2_minX <= vertex[0] <= rect2_maxX and rect2_minY <= vertex[1] <= rect2_maxY:
                    return True

            return False
        else:
            raise Exception("WTF? should never hit here")


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

        self.fold_transform = FoldTransform(start_config, end_config, self.scaff.start_time, self.scaff.end_time)

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
                if (rectangle_overlap(self_region_vertices, other_region_vertices)):
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

        basesAABB = self.scaff.aabb

        maxX = max(proj[0][0], proj[1][0], proj[2][0], proj[3][0], basesAABB[1][0])
        minX = min(proj[0][0], proj[1][0], proj[2][0], proj[3][0], basesAABB[0][0])
        maxY = max(proj[0][1], proj[1][1], proj[2][1], proj[3][1], basesAABB[1][1])
        minY = min(proj[0][1], proj[1][1], proj[2][1], proj[3][1], basesAABB[0][1])
        maxZ = max(proj[0][2], proj[1][2], proj[2][2], proj[3][2], basesAABB[1][2])
        minZ = min(proj[0][2], proj[1][2], proj[2][2], proj[3][2], basesAABB[0][2])

        if (abs(np.dot(norm, XAxis)) == 1):
            bvh.append([minX, minY, minZ])
            bvh.append([minX + self.height, maxY, maxZ])
        elif (abs(np.dot(norm, YAxis)) == 1):
            bvh.append([minX, minY, minZ])
            bvh.append([maxX, minY + self.height, maxZ])
        elif (abs(np.dot(norm, ZAxis)) == 1):
            bvh.append([minX, minY, minZ])
            bvh.append([maxX, maxY, minZ + self.height])
        else:
            raise Exception("Couldn't find fold direction when constructing BVH")

        self.bvh = np.array(bvh)

    def checkPatchAgainstBVH(self, patch):
        min = self.bvh[0]
        max = self.bvh[1]
        box = [np.array([min[0], min[1], min[2]]),
               np.array([min[0], min[1], max[2]]),
               np.array([min[0], max[1], min[2]]),
               np.array([min[0], max[1], max[2]]),
               np.array([max[0], min[1], min[2]]),
               np.array([max[0], min[1], max[2]]),
               np.array([max[0], max[1], min[2]]),
               np.array([max[0], max[1], max[2]])]

        indices = [[0, 2, 4, 6],
                   [1, 3, 5, 7],
                   [0, 1, 4, 5],
                   [2, 3, 6, 7],
                   [0, 1, 2, 3],
                   [4, 5, 6, 7]]

        for points in indices:
            face = np.array(box[points[0]], box[points[1]], box[points[2]], box[points[3]])
            if rectangle_overlap(patch.coords, face):
                return False

    def checkSelfAgainstSequenced(self, basic_scaffs: List[BasicScaff]):
        # TODO: Handle edge case
        return True

    def checkSelfAgainstUnsequenced(self, basic_scaffs: List[BasicScaff]):
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
                    return False
            else:
                base = bs.b_patch
                fold = bs.f_patch

                if type(self.scaff) == HBasicScaff:
                    if base.id == self.scaff.t_patch.id:
                        continue

                if not self.checkPatchAgainstBVH(base) or not self.checkPatchAgainstBVH(fold):
                    return False

        return True

    def checkValid(self, ownedIndex: int, mid_scaffs: List[MidScaff], folded_scaff: List[bool]) -> (bool, float):
        sequenced_scaffs = []
        unsequenced_scaffs = []
        for idx in range(0, len(mid_scaffs)):
            if idx == ownedIndex:
                continue

            for bs in mid_scaffs[idx].basic_scaffs():
                if folded_scaff[idx]:
                    sequenced_scaffs.append(bs)
                else:
                    unsequenced_scaffs.append(bs)

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
        for option_idx in range(0, len(self.fold_options)):
            if self.no_external_conflicts[option_idx]:
                continue
            else:
                self.no_external_conflicts[option_idx], cost = self.fold_options[option_idx].checkValid(ownedIndex,
                                                                                                        mid_scaffs,
                                                                                                        folded_scaff)
                if cost < min_cost:
                    min_cost = cost

        for status in self.no_external_conflicts:
            if status:
                return (True, min_cost)
        return (False, min_cost)


"""
TBasicScaff: A basic scaffold of type T
"""


class TBasicScaff(BasicScaff):
    def __init__(self, b_patch, f_patch):
        self.f_patch = f_patch
        self.b_patch = b_patch

        super().__init__()

        self.rot_axis = np.cross(f_patch.calc_normal(), b_patch.calc_normal())

    def gen_aabb(self) -> list[list[float]]:
        maxX = self.b_patch.coords[0][0]
        minX = self.b_patch.coords[0][0]
        maxY = self.b_patch.coords[0][1]
        minY = self.b_patch.coords[0][1]
        maxZ = self.b_patch.coords[0][2]
        minZ = self.b_patch.coords[0][2]

        for coord in self.b_patch.coords:
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

        return [[minX, minY, minZ], [maxX, maxY, maxZ]]

    def gen_fold_options(self, ns, nh, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        print("gen_fold_options...")
        for i in range(0, nh + 1):
            for j in range(1, ns + 1):
                for h in range(1, j + 1):
                    for k in range(0, h):
                        cost = alpha * i / nh + (1 - alpha) * ((ns - (h - k)) / ns) ** 2

                        # TODO: check this later
                        # mod = Modification(i, j, k, j - k, cost)
                        mod = Modification(i, k, h, j, cost)

                        fo_left = FoldOption(True, mod, self, self.rot_axis)
                        fo_right = FoldOption(False, mod, self, self.rot_axis)

                        # generate the fold transform from start to end?
                        # fo_left.gen_fold_transform()
                        # fo_right.gen_fold_transform()

                        # fo_left.gen_projected_region(self.rot_axis)
                        # fo_right.gen_projected_region(self.rot_axis)

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
        maxX = self.b_patch.coords[0][0]
        minX = self.b_patch.coords[0][0]
        maxY = self.b_patch.coords[0][1]
        minY = self.b_patch.coords[0][1]
        maxZ = self.b_patch.coords[0][2]
        minZ = self.b_patch.coords[0][2]

        for coord in self.b_patch.coords:
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

    def gen_fold_options(self, nh, ns, alpha):
        # Generates all possible fold solutions for TBasicScaff
        # ns: max number of patch cuts
        # nh: max number of hinges, let's enforce this to be an odd number for now
        # print("GEN FOLD OPTIONS FOR SCAFF: " + str(self.id) + "==================")

        # TODO: hard coded only doing odd number of hinges. Even numbers are too unpredictable for this purpose.
        for i in range(1, nh + 1, 2):
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

                        # generate the fold transform from start to end?
                        # fo_left.gen_fold_transform()
                        # fo_right.gen_fold_transform()

                        # fo_left.gen_projected_region(self.rot_axis)
                        # fo_right.gen_projected_region(self.rot_axis)

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
        # For now, hard code as 0 to 90 but will eventually become floats.
        self.start_time = 0
        self.end_time = 90

        self.non_conflicting_options = []

        self.best_clique = []

    def getLowCostEstimate(self):
        cost = 0
        for scaff in self.basic_scaffs:
            cost += scaff.getLowestCostEsimate()
        return cost

    def checkValid(self, ownedIndex: int, mid_scaffs: List[MidScaff], folded_scaff: List[bool]):
        cost = 0
        for bs in self.basic_scaffs:
            valid, lowest_cost = bs.checkValid(ownedIndex, mid_scaffs, folded_scaff)
            cost += lowest_cost
            if not valid:
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

    def build_execute_conflict_graph(self):
        clique, weight = self.basic_scaffs[0].getBestOption()
        self.best_clique = clique
        return clique, weight

    # Only one solution? Is there any case where this solution could impact another guy?
    # Probably not since they all fold at different times? Sequentially?
    def fold(self):
        print("TMidScaff: Implement me!")


class HMidScaff(MidScaff):
    def __init__(self, bs, nm):
        super().__init__(bs, nm)

    def gen_fold_times(self):
        print("GENERATING FOLD TIMES....")
        # If no basic scaffolds, return with error.
        if len(self.basic_scaffs) == 0:
            raise Exception("No basic scaffolds to gen fold time for!")

        # Sort basic scaffold by the position of their top patch (This only happens for HMidScaff so we can assume there is one)
        # TODO: hard coded in the Y direction, but need to generalize this
        # Sort by the Y coordinate of the top patch, which in our case should be constant for now.
        # In-place sorting should be okay for now...
        self.basic_scaffs.sort(reverse=True, key=lambda x: x.t_patch.coords[0][1])

        highest_patch = self.basic_scaffs[0].t_patch
        h0 = highest_patch.coords[0][1]

        for basic_scaff in self.basic_scaffs:
            h = basic_scaff.t_patch.coords[0][1]
            b = basic_scaff.b_patch.coords[0][1]

            # print("basic scaff:")
            # print(basic_scaff)
            basic_scaff.start_time = (h0 - h) * self.end_time / h0
            # print("start time: " + str(basic_scaff.start_time))
            basic_scaff.end_time = (h0 - b) * self.end_time / h0
            # print("end time: " + str(basic_scaff.end_time))

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

    def __init__(self, node_list, edge_list, push_dir, max_hinges, num_shrinks):
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
        self.num_shrinks = num_shrinks

        # Decomposes self and generates scaffolds
        # TODO: commented out for testing purposes, recomment back in
        # self.gen_scaffs()

    def gen_scaffs(self):
        # TODO: Di

        # generates hinge graph
        self.gen_hinge_graph()

        # generates basic scaffolds
        # self.gen_basic_scaffs()

        # generates mid-level scaffolds
        self.gen_mid_scaffs()

        # generates fold order. Fold order will be stored in self.mid_scaff_ordered
        # Midscaffs will also store the best clique  in Midscaff.best_clique
        # self.order_folds()

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
    def gen_basic_scaffs(self):
        print("gen basic scaffs")
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
                    # print(self.basic_scaffs[-1].id)
                elif len(neighbors) == 1:
                    base0 = self.node_list[neighbors[0]]
                    fold0 = self.node_list[id]
                    self.basic_scaffs.append(TBasicScaff(base0, fold0))
                    self.basic_mappings[self.basic_scaffs[-1].id] = [id, self.node_list[neighbors[0]].id]
                    # print(self.basic_scaffs[-1].id)
                else:
                    print("wtf")
        # print("end gen basic scaffs")

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
                raise Exception("Found a T scaffold, not implemented yet!")

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

    def fold(self):
        # TODO: For now, only foldabilize the first mid level scaffold.
        # TODO: Eventually I will need to do greedy one step lookahead.

        # If no scaffolds, return with an error
        if len(self.basic_scaffs) == 0:
            raise Exception("No basic scaffolds to fold")

        # First, generate time zones for basic scaffolds
        self.mid_scaffs[0].gen_fold_times()

        # First, generate basic scaffold solutions
        for scaff in self.basic_scaffs:
            # TODO: hard code alpha to be 0.5
            scaff.gen_fold_options(self.max_hinges, self.num_shrinks, 0.5)

        self.mid_scaffs[0].fold()

    def initConflictChecks(self):
        for ms in self.mid_scaffs:
            ms.initConflictChecks()

    def clearConflictChecks(self):
        for ms in self.mid_scaffs:
            ms.clearConflictChecks()

    def checkValid(self, index):
        # TODO: IMPLEMENT   
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
                    if id1 != id2 and c1 + c2 < weight1 + weight2:
                        c1 = weight1
                        c2 = weight2
                        best_Scaff = scaff1
                        bestId = id1

        print("Adding mid level scaff: " + str(bestId))

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

        while len(self.mid_scaffs_ordered) < len(self.mid_scaffs):
            self.pickNextScaff()
            self.mid_scaffs_ordered[-1].build_execute_conflict_graph()

        self.clearConflictChecks()
        self.folded_scaff = [False for i in range(size)]


# '''
# FoldManager: debug class for now, probalby won't actually use it.
# Purpose is to serve as a mini inputScaffold for now.
# '''
#
#
# class FoldManager:
#     def __init__(self):
#         self.h_basic_scaff = None  # TODO: For now a hard coded H scaffold
#         self.input_scaff = None
#
#     def generate_h_basic_scaff(self, bottom_patch: np.ndarray, fold_patch: np.ndarray, top_patch: np.ndarray):
#         # print("generate_h_basic_scaff...")
#         # print("bottom patch")
#         bPatch = Patch(bottom_patch)
#         # print("top patch")
#         tPatch = Patch(top_patch)
#         # print("fold patch")
#         fPatch = Patch(fold_patch)
#
#         scaff = HBasicScaff(fPatch, bPatch, tPatch)
#         self.h_basic_scaff = scaff
#
#     def mainFold(self, nH, nS) -> FoldOption:
#         # print("entered mainFold...")
#         # Outputs a hard coded fold option for now
#
#         # Experiment with alpha values
#         alpha = 0.5
#         cost1 = alpha * 0 / 1 + (1 - alpha) / 1
#         mod1 = Modification(nH, 0, 3, nS, cost1)
#         # patch_list = [self.h_basic_scaff.f_patch, self.h_basic_scaff.b_patch, self.h_basic_scaff.t_patch]
#         fo = FoldOption(False, mod1, self.h_basic_scaff, XAxis)
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
    input.fold()

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
    input.fold()

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
    input.fold()

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

    input = InputScaff(nodes, edges, push_dir, 5, 3)

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


test_cube_shape()


def test_input_scaff():
    coords1 = np.array([(0, 3, 0), (1, 3, 0), (1, 3, 1), (0, 3, 1)])  # base 0
    coords2 = np.array([(0, 2, 0), (1, 2, 0), (1, 2, 1), (0, 2, 1)])  # base 1
    coords3 = np.array([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)])  # base 2
    coords4 = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])  # base 3

    coords5 = np.array([(0.5, 0, 1), (0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1)])  # fold 0
    coords6 = np.array([(0.5, 1, 1), (0.5, 1, 0), (0.5, 2, 0), (0.5, 2, 1)])  # fold 1
    coords7 = np.array([(0.5, 2, 1), (0.5, 2, 0), (0.5, 3, 0), (0.5, 3, 1)])  # fold 2

    b1 = Patch(coords1)
    b2 = Patch(coords2)
    b3 = Patch(coords3)
    b4 = Patch(coords4)

    f1 = Patch(coords5)
    f2 = Patch(coords6)
    f3 = Patch(coords7)

    b1.id = 0
    b2.id = 1
    b3.id = 2
    b4.id = 3

    f1.id = 4
    f2.id = 5
    f3.id = 6

    nodes = [b1, b2, b3, b4, f1, f2, f3]
    edges = [[4, 0], [5, 1], [6, 2], [3, 6], [2, 5], [1, 4]]

    push_dir = YAxis

    input = InputScaff(nodes, edges, push_dir, 7, 4)

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


def t_construct_bvh():
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
    input.basic_scaffs[0].gen_fold_options(3, 2, 0.5)

    # print every fold option's bvh
    for basic_scaff in input.basic_scaffs:
        for fold_option in basic_scaff.fold_options:
            print("BVH")
            print(fold_option.bvh)

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

# t_construct_bvh()
