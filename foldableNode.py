from __future__ import annotations

import copy
import os
import sys
import random
import math

import maya.OpenMaya as OpenMaya
import maya.OpenMayaAnim as OpenMayaAnim
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds
import foldMain as fold
import numpy as np
from typing import Dict, List, Set


# Useful functions for declaring attributes as inputs or outputs.
def MAKE_INPUT(attr):
    attr.setKeyable(True)
    attr.setStorable(True)
    attr.setReadable(True)
    attr.setWritable(True)


def MAKE_OUTPUT(attr):
    attr.setKeyable(False)
    attr.setStorable(False)
    attr.setReadable(True)
    attr.setWritable(False)


# Define the name of the node
kPluginNodeTypeName = "foldableNode"

# Give the node a unique ID. Make sure this ID is different from all of your
# other nodes!
foldableNodeId = OpenMaya.MTypeId(0x8706)

EPS = 0.005

def resetFoldClass():
    fold.InputScaff.id_incr = 0
    fold.BasicScaff.id_incr = 0
    fold.Patch.id_incr = 0

# Static helper functions
def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)
    return OpenMaya.MFnTransform(transform_dag_path)


def getObjectObjectFromDag(name: str) -> OpenMaya.MDagPath:
    # Create an MSelectionList object to store the plane name
    print("getting dag path for: {}".format(name))
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)

    print("returning transform dag path..")
    return transform_dag_path


def setUpVertBasicScene():
    print("setUpVertScene: BASIC")
    pFold = getObjectTransformFromDag("pFoldH")
    pBaseTop = getObjectTransformFromDag("pBaseTopH")
    pBaseBottom = getObjectTransformFromDag("pBaseBottomH")

    # Set the translation for pBottom to 0, 0, 0 and pTop to 0, 1, 0.
    pFold.setTranslation(OpenMaya.MVector(0, 0.5, 5), OpenMaya.MSpace.kWorld)
    pBaseTop.setTranslation(OpenMaya.MVector(0, 1, 5), OpenMaya.MSpace.kWorld)
    pBaseBottom.setTranslation(OpenMaya.MVector(0, 0, 5), OpenMaya.MSpace.kWorld)

    # # Set the rotation for both to 0, 0, -90. Keep in mind that EulerRotation is in radians.
    pFold.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pBaseTop.setRotation(OpenMaya.MEulerRotation(0, 0, 0))
    pBaseBottom.setRotation(OpenMaya.MEulerRotation(0, 0, 0))


# Returns a dictionary of names and positions in world space.
def getObjectVerticeNamesAndPositions(name: str) -> Dict[str, List[float]]:
    # TODO: There are probably better ways by getting the vertex iterator.
    vertex_count = cmds.polyEvaluate(name, vertex=True)
    vertices = {}
    for i in range(vertex_count):
        vertex_name = "{}.vtx[{}]".format(name, i)
        vertex_translation = cmds.pointPosition(vertex_name, world=True)

        # print
        # print("vertex_name: {}".format(vertex_name))
        # print("vertex_translation: {}".format(vertex_translation))

        vertices[vertex_name] = vertex_translation

    return vertices


# Write a function that for a list of vertices, returns the closest n vertices to a given point p.
# The function should return a list of triplet (vertex_name, distance, vertex_position).
# The list should be sorted by distance from p.
def getClosestVertices(vertices: dict, p: OpenMaya.MVector, n: int) -> list:
    # Iterate through the vertices and calculate the distance from p.
    # Store the distance and vertex name in a list.
    # Sort the list by distance.
    # Return the first n elements of the list.
    distList = []
    for vertex_name, vertex_position in vertices.items():
        vertexPoint = OpenMaya.MVector(vertex_position[0], vertex_position[1], vertex_position[2])
        dist = OpenMaya.MVector(p - vertexPoint).length()
        distList.append((vertex_name, dist, vertexPoint))

    distList.sort(key=lambda x: x[1])
    return distList[:n]

def getClosestVerticesTopBase(vertices: dict, base: str, n: int) -> list:
    print("finding closest vertices top base")

    transformDagPath = getObjectObjectFromDag(base)

    # Get the shape node
    # print("Getting shape node")
    transformDagPath.extendToShape()

    # print("Checking if shape node is of type mesh")
    # Check if the shape node is of type "mesh"
    if transformDagPath.node().hasFn(OpenMaya.MFn.kMesh):
        # print("creating mesh")
        # Create an MFnMesh function set
        fnMesh = OpenMaya.MFnMesh(transformDagPath)
        distList = []
        for vertex_name, vertex_position in vertices.items():
            vertexPoint = OpenMaya.MPoint(vertex_position[0], vertex_position[1], vertex_position[2])
            closestPointOnPlane = OpenMaya.MPoint()

            fnMesh.getClosestPoint(vertexPoint, closestPointOnPlane, OpenMaya.MSpace.kWorld)

            dist = OpenMaya.MVector(vertexPoint - closestPointOnPlane).length()
            distList.append((vertex_name, dist, OpenMaya.MVector(closestPointOnPlane)))

        distList.sort(key=lambda x: x[1])
        return distList[:n]


def checkScaffoldConnection(pivot: OpenMaya.MVector, middlepoint: OpenMaya.MVector):
    dist = OpenMaya.MVector(pivot - middlepoint).length()
    # print("Pivot distance to middle point: {:.6f}".format(dist))
    if dist > 0.0001:
        print("Error: Distance is not 0. Patches are not connected")
        exit(1)


# Need to keep this function because it is a special condition that Idk how to fix yet
def checkScaffoldConnectionTopBase(parent, childPatch: str):
    childVertices = getObjectVerticeNamesAndPositions(childPatch)

    # Get child's global Y position
    # TODO: make this more generic in the future
    childY = float(childVertices["{}.vtx[0]".format(childPatch)][1])

    # Get child's max x and min x
    childMaxX = max(childVertices.values(), key=lambda x: x[0])[0]
    childMinX = min(childVertices.values(), key=lambda x: x[0])[0]

    # Get child's max z and min z
    childMaxZ = max(childVertices.values(), key=lambda x: x[2])[2]
    childMinZ = min(childVertices.values(), key=lambda x: x[2])[2]

    # check if both of the parent's vertices are within the child's bounding box
    connected = True
    for element in parent:
        vertex = element[2]
        # print("vertex: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))
        if abs(vertex[1] - childY) > 0.0001:
            print("Y values are not the same!")
            print("Parent Y: {}".format(vertex[1]))
            print("Child Y: {}".format(childY))
            connected = False
            break
        if vertex[0] < childMinX - EPS:
            print("X value is less than minX")
            connected = False
            break
        if vertex[0] > childMaxX + EPS:
            print("X value is larger than maxX")
            connected = False
            break
        if vertex[2] < childMinZ - EPS:
            print("Z value is smaller childMinZ")
            connected = False
            break
        if vertex[2] > childMaxZ + EPS:
            print("Z value is larger than childMaxZ")
            connected = False
            break

    if not connected:
        print("Error TOPBASE: Patches are not connected")
        exit(1)


def checkScaffoldConnectionBaseNoErr(base: str, foldable) -> bool:
    # Check scaffold connection between a base patch and closest points on the foldable patch
    baseVertices = getObjectVerticeNamesAndPositions(base)

    # Get child's global Y position
    # TODO: make this more generic in the future
    baseY = float(baseVertices["{}.vtx[0]".format(base)][1])

    # Get child's max x and min x
    baseMaxX = max(baseVertices.values(), key=lambda x: x[0])[0]
    baseMinX = min(baseVertices.values(), key=lambda x: x[0])[0]

    # Get child's max z and min z
    baseMaxZ = max(baseVertices.values(), key=lambda x: x[2])[2]
    baseMinZ = min(baseVertices.values(), key=lambda x: x[2])[2]

    # check if both of the parent's vertices are within the child's bounding box
    connected = True
    for element in foldable:
        vertex = element[2]
        # print("verices in foldable closest to base: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))
        if abs(vertex[1] - baseY) > 0.0001:
            print("Y values are not the same!")
            print("Parent Y: {}".format(vertex[1]))
            print("Child Y: {}".format(baseY))
            connected = False
            break
        if (vertex[0] < baseMinX - EPS):
            print("X value is less than minX")
            print("X value: {}".format(vertex[0]))
            print("minX: {}".format(baseMinX))
            connected = False
            break
        if vertex[0] > baseMaxX + EPS:
            print("X value is larger than maxX")
            connected = False
            break
        if vertex[2] < baseMinZ - EPS:
            print("Z value is smaller childMinZ")
            connected = False
            break
        if vertex[2] > baseMaxZ + EPS:
            print("Z value is larger than childMaxZ")
            connected = False
            break
    return connected


def isPolyPlane(obj):
    # Create an MSelectionList object
    transformDagPath = getObjectObjectFromDag(obj)

    # Get the shape node
    # print("Getting shape node")
    transformDagPath.extendToShape()

    # print("Checking if shape node is of type mesh")
    # Check if the shape node is of type "mesh"
    if transformDagPath.node().hasFn(OpenMaya.MFn.kMesh):
        # print("creating mesh")
        # Create an MFnMesh function set
        fnMesh = OpenMaya.MFnMesh(transformDagPath)

        # Get the number of faces in the mesh
        numFaces = fnMesh.numPolygons()

        # If the mesh has only one face, it can be considered a polygonal plane
        if numFaces == 1:
            return True
    return False


# TODO: bring a lot of the functions into a parent class
class MayaHBasicScaffoldWrapper():

    def __init__(self, patchObjects: List[fold.Patch], basePatch: str, patches: List[str], pushAxis: OpenMaya.MVector, maxHinges: int, shrinks: int):
        self.basePatch = basePatch
        self.patchesObjs = patches
        self.patches = patches

        # TODO: hard coded for now but make it dynamic later
        # Get the scaleX of patches[0]
        self.origFoldPatchHeight = cmds.getAttr("{}.scaleX".format(patches[0]))

        self.pushAxis = pushAxis

        self.maxHinges = maxHinges
        self.shrinks = shrinks

        self.newShapes = []

        self.inInitialPatches = []
        self.shapeTraverseOrder: List[str] = []
        self.shapeResetTransforms: Dict[str, List[List[float]]] = {}

        self.shapeOriginalTransforms: Dict[str, List[List[float]]] = {}

        self.additionalTransforms: Dict[str, List[float]] = {}

        # # TODO:remove later
        # self.foldManagerOption = None
        # self.foldManager = None

        '''
        Basic scaffold peek:
        - basePatches: the base patch
        - foldablePatch: foldable patch
        - foldOption: list of fold solutions
            - foldTransform: 
                - startAngles
                - endAngles
                - startTime
                - endTime
            - Modifcation
            - isLeft
            - rotationalAxis

        '''
        # TODO: currently basicScaffolds are being created by the inputScaffold manually for testing purposes.
        # patchVerts = np.array(self.getAllPatchVertices())
        # bottomPatch = fold.Patch(patchVerts[0])
        # foldPatch = fold.Patch(patchVerts[1])
        # topPatch = fold.Patch(patchVerts[2])
        self.basicScaffold: fold.HBasicScaff = fold.HBasicScaff(patchObjects[0], patchObjects[1], patchObjects[2])

        # Parent/Child relationships
        # TODO: can get away with only one parent for now
        self.parent = None
        self.children = []

    def setParent(self, parent: MayaHBasicScaffoldWrapper):
        self.parent = parent

    def addChild(self, child: MayaHBasicScaffoldWrapper):
        self.children.append(child)

    def getBasePatch(self) -> str:
        return self.basePatch

    # TODO: Test moving it to inputScaffold
    def getPatches(self) -> List[str]:
        return self.patches

    def getPatchesIncludeBase(self) -> List[str]:
        patches = [self.basePatch]
        patches.extend(self.patches)
        return patches

    # TODO: why is this a list list list?
    def getAllPatchVertices(self) -> List[List[List[float]]]:
        # For each element in self.getPatchesIncludeBase, call getObjectVerticeNamesAndPositions
        shapeVertices = []
        for patch in self.getPatchesIncludeBase():
            vertices = getObjectVerticeNamesAndPositions(patch)
            shapeVertices.append(list(vertices.values()))

        return shapeVertices

    def cleanUpSplitPatches(self):
        print("cleaning up split patches...")
        for i in range(0, len(self.newShapes)):
            cmds.delete(self.newShapes[i])

        # Clear the new shapes list.
        self.newShapes = []

    # When a parent scaffold translates, the child scaffold must also translate with it on every reset to treat it
    # as the new origin
    def translateWithParentScaff(self, translateVector: OpenMaya.MVector):
        print("Translating with parent scaff: " + str(self.basicScaffold.id))
        if (len(self.shapeOriginalTransforms) == 0):
            raise Exception("No original transforms in the scaffol!")

        for shapeKey in self.shapeOriginalTransforms.keys():
            # Get the original translation
            original_translate = self.shapeOriginalTransforms[shapeKey][0]

            # if key is in additionalTransforms, add the new translateVec to the additionalTransforms
            if shapeKey in self.additionalTransforms:
                self.additionalTransforms[shapeKey][0] += translateVector.x
                self.additionalTransforms[shapeKey][1] += translateVector.y
                self.additionalTransforms[shapeKey][2] += translateVector.z
            else:
                self.additionalTransforms[shapeKey] = [translateVector.x, translateVector.y, translateVector.z]

            additionalTransform = OpenMaya.MVector(self.additionalTransforms[shapeKey][0],
                                                   self.additionalTransforms[shapeKey][1],
                                                   self.additionalTransforms[shapeKey][2])

            translate_vec = OpenMaya.MVector(original_translate[0], original_translate[1], original_translate[2])
            translate_vec += OpenMaya.MVector(additionalTransform[0], additionalTransform[1], additionalTransform[2])
            # Replace the translation with the new translation

            print("translating by: {:.6f}, {:.6f}, {:.6f}".format(translateVector.x,
                                                                  translateVector.y,
                                                                    translateVector.z))

            print("after translating position: {:.6f}, {:.6f}, {:.6f}".format(translate_vec.x,
                                                                              translate_vec.y,
                                                                              translate_vec.z))

            # cmds.setAttr(shapeKey + ".translate", translate_vec[0], translate_vec[1], translate_vec[2])
            # transform = getObjectTransformFromDag(shapeKey)
            # transform.translateBy(translate_vec, OpenMaya.MSpace.kWorld)

            self.shapeResetTransforms[shapeKey][0] = [translate_vec.x, translate_vec.y, translate_vec.z]

        if (len(self.children) == 0):
            print("Scaffold " + str(self.basicScaffold.id) + " has no children=============================")

        for child in self.children:
            child.translateWithParentScaff(translateVector)

    def setUpGenericScene(self, upperPatches: List[str]):
        print("Setting up Generic scene...")
        # TODO: theoretically we should only need to move things in the upper patches
        # Get the transforms for each item in upper patches

        self.additionalTransforms = {}

        transforms = []
        for patch in upperPatches:
            transforms.append(getObjectTransformFromDag(patch))

        original_transforms = self.shapeResetTransforms

        # Set the translation for each of the patches to the original translations
        for i in range(0, len(upperPatches)):
            patch_name = upperPatches[i]
            original_translate = original_transforms[patch_name][0]
            translate_vec = OpenMaya.MVector(original_translate[0], original_translate[1], original_translate[2])
            transforms[i].setTranslation(translate_vec, OpenMaya.MSpace.kWorld)

        # Set the rotation for each of the patches to the original rotations
        for i in range(0, len(transforms)):
            patch_name = upperPatches[i]
            original_rotate = original_transforms[patch_name][1]
            cmds.setAttr(patch_name + ".rotate", original_rotate[0][0], original_rotate[0][1], original_rotate[0][2])

    def restoreInitialState(self):
        # Sets the shapeTraverseOrder to the original scaff's patches
        # Sets
        # print("Restoring Initial State...")
        self.shapeTraverseOrder = self.getPatchesIncludeBase()

        # Clears self.shapeResetTransforms
        self.shapeResetTransforms = {}
        self.additionalTransforms = {}

        # fill new_translation and new_rotation with original values
        for shape in self.shapeTraverseOrder:
            transform = getObjectTransformFromDag(shape)
            translate = transform.translation(OpenMaya.MSpace.kWorld)
            rotate = cmds.getAttr(shape + ".rotate")

            # Reset self.shapeResetTransforms
            self.shapeResetTransforms[shape] = [[translate[0], translate[1], translate[2]], rotate]

        # ShapeOriginalTransforms will be used to calculate shapeResetTransforms when parent scaffolds move
        self.shapeOriginalTransforms = copy.deepcopy(self.shapeResetTransforms)

        # Reset back to original shapes so you can break them again
        self.cleanUpSplitPatches()

    def shrinkPatch(self, shapeTraverseOrder, endPiece, numPieces, startPiece, rotAxis):
        # print("shrinking patches...")

        # TODO: shrink only the middle patch
        # Shrink the patches except first and last
        for i in range(1, len(shapeTraverseOrder) - 1):
            foldable_patch = shapeTraverseOrder[i]
            middle = 1 / 2  # hard coded for now

            # Translate patch to the new midpoint
            pieceWidth = 1.0 / numPieces  # hard coded for now
            newMiddle = (startPiece + endPiece) * pieceWidth / 2

            # print("newMiddle in Z direction: {}".format(newMiddle))
            transform = getObjectTransformFromDag(foldable_patch)

            shrinkAxis = OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2])

            translation = shrinkAxis * newMiddle - shrinkAxis * middle
            # print("translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1], translation[2]))
            transform.translateBy(translation, OpenMaya.MSpace.kTransform)

            # Shrink patch by numPieces in the hard coded z direction
            shrinkFactor = (endPiece - startPiece) / numPieces
            cmds.setAttr(foldable_patch + ".scaleZ", shrinkFactor)

    def generateNewPatches(self, originalPatch: str, numHinges: int) -> (List[str], List[List[List[float]]]):

        # Compute the new patch scale values based on original_patch's scale and num_patches
        # TODO: Hard coded for split in the x Direction, but need to be more general later on.
        numPatches = numHinges + 1
        originalScaleX = cmds.getAttr(originalPatch + ".scaleX")
        originalScaleZ = cmds.getAttr(originalPatch + ".scaleZ")

        newPatchScale = originalScaleX / numPatches

        # Generate new patches.
        newPatches = []
        for i in range(0, numPatches):
            # TODO: Based on the axis we shrink, either width or height will be the original patch's scale
            # This command generates a new polyplane in the scene
            newPatch = cmds.polyPlane(name=originalPatch + "_" + str(i), width=newPatchScale, height=originalScaleZ,
                                      subdivisionsX=1,
                                      subdivisionsY=1)
            newPatches.append(newPatch[0])

        # Rotate the new patches with the same rotation as the original_patch
        originalRotation = cmds.getAttr(originalPatch + ".rotate")
        for i in range(0, len(newPatches)):
            cmds.setAttr(newPatches[i] + ".rotate", originalRotation[0][0], originalRotation[0][1],
                         originalRotation[0][2])

        # Translate the patches along the direction it has been scaled in (but that is local)
        # TODO: Axis of scaling is hard coded
        originalTranslation = cmds.getAttr(originalPatch + ".translate")

        # Get the world location of the bottom of the original patch
        # TODO: hard coded for the Y direction
        originalPatchBottom = originalTranslation[0][1] - originalScaleX * 0.5
        newPatchPositions = []
        newTransforms = []
        for i in range(0, len(newPatches)):
            newTranslate = [originalTranslation[0][0], originalPatchBottom + newPatchScale * (i + 0.5),
                            originalTranslation[0][2]]
            newPatchPositions.append(newTranslate)
            cmds.setAttr(newPatches[i] + ".translate", newTranslate[0], newTranslate[1], newTranslate[2])

            # Append new patch transform to list of new transforms
            # Which will be used for its scene reset at the beginning
            # TODO: why is this a tuple with the name in the first spot?
            newTransforms.append([newTranslate, originalRotation])

        # Pivot the patches.
        for i in range(0, len(newPatches)):
            # Set the pivot location to the bottom of the patch
            newPivot = [newPatchScale * 0.5, 0, 0]
            transform = getObjectTransformFromDag(newPatches[i])
            transform.setRotatePivot(OpenMaya.MPoint(newPivot[0], newPivot[1], newPivot[2]), OpenMaya.MSpace.kTransform,
                                     True)

        newPatches.reverse()
        newTransforms.reverse()

        return newPatches, newTransforms

    def breakPatches(self, shapeTraverseOrder: List[str], numHinges: int):

        # Render the original foldable patch invisible
        # Take every guy except the last guy and hide it
        # TODO: now, i know that since this is a basic patch, there should only ever be one of these guys.
        foldablePatches = self.patches[:-1]
        for patch in foldablePatches:
            cmds.setAttr(patch + ".visibility", False)

        # print("break patches called")
        # Break every patch except the last one
        for j in range(1, len(shapeTraverseOrder) - 1):  # every patch except last guy is foldable
            foldablePatch = shapeTraverseOrder[
                j]  # TODO: make more generic, currently assumes foldable patch is at the center

            # Remove the foldable patch from the shape_traverse_order and delete its transforms
            shapeTraverseOrder.remove(foldablePatch)
            del self.shapeResetTransforms[foldablePatch]
            del self.shapeOriginalTransforms[foldablePatch]

            # TODO: diabolical fix it
            if (foldablePatch in self.additionalTransforms):
                del self.additionalTransforms[foldablePatch]

            newPatches, newTransforms = self.generateNewPatches(foldablePatch, numHinges)

            # Add the new patch transforms to the shape_reset_transforms and insert new patches to shape_traverse_order
            for i in range(0, len(newPatches)):
                shapeTraverseOrder.insert(j, newPatches[i])
                self.shapeResetTransforms[newPatches[i]] = [newTransforms[i][0], newTransforms[i][1]]
                self.shapeOriginalTransforms[newPatches[i]] = [newTransforms[i][0], newTransforms[i][1]]

                # Keep track of the new patches just created so we can delete it on the next iteration
                self.newShapes.append(newPatches[i])

    def getPatchPivots(self, shapeTraverseOrder: List[str]) -> List[OpenMaya.MPoint]:
        patchPivots = []
        for shape in shapeTraverseOrder:
            pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
            patchPivots.append(pivot)
        return patchPivots

    def findClosestMidpointsOnPatches(self, patchPivots: List[OpenMaya.MPoint], shapeTraverseOrder: List[str]) -> (
            List[List], List[float]):
        # print("finding cloest midpoints on patches... ===========")
        closestVertices = []
        midPoints = []

        for i in range(0, len(shapeTraverseOrder) - 1):
            # For each parent patch, get their vertices.
            shape = shapeTraverseOrder[i]
            child = shapeTraverseOrder[i + 1]

            bottomVertices = getObjectVerticeNamesAndPositions(shape)

            childPivot = patchPivots[i + 1]

            # find two vertices that are closest to childPivot. Print their name, location, and distance.
            vertId = len(closestVertices)
            if (i == len(shapeTraverseOrder) - 2):
                currentClosest = getClosestVerticesTopBase(bottomVertices, child, 2)
            else:
                # TODO: figure out why commenting the top line ends up shifting a scaffold for nor eaosn
                # currentClosest = getClosestVerticesTopBase(bottomVertices, child, 2)
                currentClosest = getClosestVertices(bottomVertices, childPivot, 2)

            closestVertices.append(currentClosest)
            print("Current Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(currentClosest[0][0],
                                                                                             currentClosest[0][1],
                                                                                             currentClosest[0][2][0],
                                                                                             currentClosest[0][2][1],
                                                                                             currentClosest[0][2][2]))
            print("Current Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(currentClosest[1][0],
                                                                                                currentClosest[1][1],
                                                                                                currentClosest[1][2][0],
                                                                                                currentClosest[1][2][1],
                                                                                                currentClosest[1][2][2]))

            # print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(closestVertices[vertId][0][0],
            #                                                                           closestVertices[vertId][0][1],
            #                                                                           closestVertices[vertId][0][2][0],
            #                                                                           closestVertices[vertId][0][2][1],
            #                                                                           closestVertices[vertId][0][2][2]))
            # print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(closestVertices[vertId][1][0],
            #                                                                           closestVertices[vertId][1][1],
            #                                                                           closestVertices[vertId][1][2][0],
            #                                                                           closestVertices[vertId][1][2][1],
            #                                                                           closestVertices[vertId][1][2][2]))

            # Get the middle point between the two vertices.
            verticeDist = closestVertices[vertId][0][2] + closestVertices[vertId][1][2]
            # print("Vertice Dist: {:.6f}, {:.6f}, {:.6f}".format(verticeDist[0], verticeDist[1], verticeDist[2]))
            middlePoint = (verticeDist * 0.5)
            # print("Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePoint[0], middlePoint[1], middlePoint[2]))

            midPoints.append(middlePoint)

            # Ensure the parent and child are actually connected
            # TODO: generalize to T scaffolds as well
            if (i == 0):
                # TODO: should probably eventually check but bypass for now
                continue
            if (i == len(shapeTraverseOrder) - 2):
                # TODO: generalize to the one without the error
                checkScaffoldConnectionTopBase(currentClosest, child)
            else:
                checkScaffoldConnection(childPivot, middlePoint)

        return closestVertices, midPoints

    def getPatchTransforms(self, shapeTraverseOrder: List[str]) -> List[OpenMaya.MFnTransform]:
        patchTransforms = []
        for shape in shapeTraverseOrder:
            pTransform = getObjectTransformFromDag(shape)
            patchTransforms.append(pTransform)
        return patchTransforms

    def computeAngle(self, endAngles, endTime, numHinges, startTime, t):
        targetPatchHeight = (endTime - t) / (endTime - startTime) * self.origFoldPatchHeight
        rightTriangleHeight = targetPatchHeight / (numHinges + 1)
        rightTriangleHypotenuse = self.origFoldPatchHeight / (numHinges + 1)
        asin = math.asin(rightTriangleHeight / rightTriangleHypotenuse)
        angle = endAngles[0] - math.degrees(asin)
        # print("angle: " + str(angle))
        return angle

    def rotatePatches(self, angle: float, rotAxis: List[float], shapeTraverseOrder: List[str], isLeft: bool):
        if not isLeft:
            angle = -angle

        for i in range(1, len(shapeTraverseOrder) - 1):
            shape = shapeTraverseOrder[i]
            pTransform = getObjectTransformFromDag(shape)

            q = OpenMaya.MQuaternion(math.radians(angle), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
            pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)

            angle = -angle

    def updatePatchTranslations(self, closestVertices: List, midPoints: List, patchPivots: List, patchTransforms: List,
                                shapeTraverseOrder: List[str]):
        # Get the new closest vertices without changing the original closest vertices
        newClosestVertices = closestVertices.copy()
        for i in range(0, len(patchPivots) - 1):
            # Obtain child pivot so we can use it later for translation
            for j in range(0, len(newClosestVertices[
                                      i])):  # index and use information from updated vertex positions. There should only be 2 verts here
                vertex_name, dist, vertexPoint = newClosestVertices[i][j]

                # Get the world position of the vertex and convert it to an MVector
                vertexPoint = cmds.pointPosition(vertex_name, world=True)
                vertexPoint = OpenMaya.MVector(vertexPoint[0], vertexPoint[1], vertexPoint[2])

                # print("Vertex Point: {:.6f}, {:.6f}, {:.6f}".format(vertexPoint[0], vertexPoint[1], vertexPoint[2]))

                newClosestVertices[i][j] = (
                    vertex_name, 0,  # not sure if dist is important anymore
                    vertexPoint)

                # Print new location and distance.
                # print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(newClosestVertices[i][j][0],
                #                                                                           newClosestVertices[i][j][1],
                #                                                                           newClosestVertices[i][j][2][
                #                                                                               0],
                #                                                                           newClosestVertices[i][j][2][
                #                                                                               1],
                #                                                                           newClosestVertices[i][j][2][
                #                                                                               2]))

            # Midpoint formula to solve for the midpoint betwen the two closest vertices.
            verticeDistNew = newClosestVertices[i][0][2] + newClosestVertices[i][1][2]
            middlePointNew = (verticeDistNew * 0.5)
            # print(
            #     "Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePointNew[0], middlePointNew[1], middlePointNew[2]))

            # Get the translation from the old middle point to the new middle point.
            ogMidPoint = midPoints[i]
            translation = middlePointNew - ogMidPoint
            # print("Middle point translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1],
            #                                                                 translation[2]))

            # Translate child patch by the translation.
            # print("TRANSLATING THE CHILD PATCH CALLED: " + shapeTraverseOrder[i + 1])
            childPatchTransform = patchTransforms[i + 1]

            # print the childPatch's transform before translation
            print("SHAPE: " + shapeTraverseOrder[i + 1])
            allVertices = list(getObjectVerticeNamesAndPositions(shapeTraverseOrder[i + 1]).values())
            for vertex in allVertices:
                print("PRE Vertex Point: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))

            print("Translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1], translation[2]))
            childPatchTransform.translateBy(translation, OpenMaya.MSpace.kWorld)

            # print the childPatch's transform afer translation
            allVertices = list(getObjectVerticeNamesAndPositions(shapeTraverseOrder[i + 1]).values())
            for vertex in allVertices:
                print("POST Vertex Point: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))

            if (i == len(shapeTraverseOrder) - 2):
                # If we are on the second to last patch, then we updated the top patch's location
                # and we need to update the children's reset locations.
                for child in self.children:
                    child.translateWithParentScaff(translation)


    # Splits the foldTest function into two parts.
    def foldKeyframe(self, time, shapeTraverseOrder: List[str], foldSolution: fold.FoldOption, recreatePatches: bool,
                     startTime: int, endTime: int):
        # Get the relevant information from the fold_solution
        startAngles = foldSolution.fold_transform.startAngles  # TODO: lowkey aren't start angles always 0??
        endAngles = foldSolution.fold_transform.endAngles

        isLeft = foldSolution.isleft

        # Hinge variables
        numHinges = foldSolution.modification.num_hinges

        # Shrinking variables
        startPiece = foldSolution.modification.range_start
        endPiece = foldSolution.modification.range_end  # non inclusive
        numPieces = foldSolution.modification.num_pieces

        t = time  # dictate that the end time is 90 frames hard coded for now

        print("time: " + str(t))

        # TODO: make more generic in the future
        rotAxis = self.basicScaffold.rot_axis

        # Update the list of shape_traverse_order to include the new patches where the old patch was
        if (recreatePatches and numHinges > 0):
            self.breakPatches(shapeTraverseOrder, numHinges)

        # TODO: let author dictate end time.. but this doesn't realy matter.
        # TODO: THIS FORMULA DOES NOT WORK

        # Moved from the recreate_patches condition because we always want this to be visible if no hinges
        # TODO: ventually move this to a better place
        if (numHinges == 0):
            # set middle patch to be visible
            cmds.setAttr(shapeTraverseOrder[0] + ".visibility", 1)

        # Loop through the patches and get all of their pivots.
        patchPivots = self.getPatchPivots(shapeTraverseOrder)

        # Find the closest vertices to the patch pivots and calculate the midPoints, also check scaff is connected
        closestVertices, midPoints = self.findClosestMidpointsOnPatches(patchPivots, shapeTraverseOrder)

        # Compute angle
        if (endTime > t >= startTime):
            angle = self.computeAngle(endAngles, endTime, numHinges, startTime, t)
        elif (t < startTime):
            return
        else:
            angle = endAngles[0]

        # Get the transforms of the patches
        patchTransforms = self.getPatchTransforms(shapeTraverseOrder)

        # Perform rotations at once, but do not rotate the last patch
        self.rotatePatches(angle, rotAxis, shapeTraverseOrder, isLeft)

        # Update location of closest vertices after rotation and update children translations
        self.updatePatchTranslations(closestVertices, midPoints, patchPivots, patchTransforms, shapeTraverseOrder)

        # Has to go at the end or something otherwise you'll get a space between top patch and the folds
        self.shrinkPatch(shapeTraverseOrder, endPiece, numPieces, startPiece, rotAxis)

    # Fold test for non hard coded transforms: Part 1 of the logic from foldTest, calls foldKeyframe()
    # AT this point should already have the best fold option
    def foldAnimateBasic(self, time: int, recreatePatches: bool):
        foldOption = self.basicScaffold.optimal_fold_option
        startTime = foldOption.fold_transform.start_time
        endTime = foldOption.fold_transform.end_time

        self.inInitialPatches = self.getPatchesIncludeBase()

        if (len(self.shapeTraverseOrder) == 0 or recreatePatches):
            self.restoreInitialState()

        else:
            # Reset the scene
            # TODO; Might not work anymore in a bit
            self.setUpGenericScene(self.shapeTraverseOrder)

        # Call the keyframe funtion but with the LOCAL TIME rather than the current global time
        # TODO: get rid of these params since some of them are just member vars
        self.foldKeyframe(time, self.shapeTraverseOrder, foldOption, recreatePatches, startTime, endTime)


class MayaInputScaffoldWrapper():
    def __init__(self, patches: List[str], pushAxis: OpenMaya.MVector, nH: int, nS: int):
        self.pushAxis: OpenMaya.MVector = pushAxis
        self.patches: List[str] = patches

        # Both of the following are used purely for the InputScaffold's initialization
        # List of patch objects, same as patches, but not strings
        self.patchesObjs: List[fold.Patch] = []

        # Patches split into foldable and base patches
        self.basesObjs: List[fold.Patch] = []
        self.foldablesObjs: List[fold.Patch] = []

        # List of edges, using patch Objecs
        self.edgesObjs: List[List[fold.Patch]] = []

        self.maxHinges = nH
        self.shrinks = nS

        self.inputScaffold = None

        # Should be a list of MayaHBasicScaffoldWrapper or MayaTBasicScaffoldWrapper
        self.basicScaffoldWrappers: List = []

    def getAllPatches(self) -> List[str]:
        return self.patches

    def addBasicScaffold(self, basicScaffoldWrapper: MayaHBasicScaffoldWrapper):
        self.basicScaffoldWrappers.append(basicScaffoldWrapper)
        self.inputScaffold.basic_scaffs.append(basicScaffoldWrapper.basicScaffold)

    def genConnectivityInfo(self):
        """
        Test each patch for connectivity to other patches
        First, only get the patches that are normal to the pushing direction called base patch
        For each base patch, test for connectivity against all other patches (foldable patches)
        If they are close enough to each other via check-scaffold connectivity, then add an edge between them
        """

        print("Generating connectivity Info...")
        for patch in self.patches:
            # Create a patch object and add it to self.patchesObjs
            patchVertices = np.array(list(getObjectVerticeNamesAndPositions(patch).values()))
            patchObj = fold.Patch(patchVertices, patch)
            self.patchesObjs.append(patchObj)

            # Get the surface normal of the patch in world space
            planeDagPath = getObjectObjectFromDag(patch)
            fnMesh = OpenMaya.MFnMesh(planeDagPath)

            # Get the normal of the plane's first face (face index 0)
            normal = OpenMaya.MVector()

            # Apparently the normal agument is the SECOND argument in this dumbass function
            fnMesh.getPolygonNormal(0, normal, OpenMaya.MSpace.kWorld)

            # Get the dot product of normal and pushDir
            dot = self.pushAxis * normal
            if (abs(abs(dot) - 1) < 0.0001):
                # Parallel
                self.basesObjs.append(patchObj)
            else:
                self.foldablesObjs.append(patchObj)

        edgesObjs = []

        # For every base in basePatches, test for connectivity with every foldable_patch
        for baseObj in self.basesObjs:
            for foldpatchObj in self.foldablesObjs:
                # Find pivot of base
                print("Checking connectivity between " + baseObj.name + " AND " + foldpatchObj.name)
                pivot = getObjectTransformFromDag(baseObj.name).rotatePivot(OpenMaya.MSpace.kWorld)

                # find the closest vertices from fold to pivot
                vertices = getObjectVerticeNamesAndPositions(foldpatchObj.name)
                closestVertices = getClosestVertices(vertices, pivot, 2)
                # TODO: figure out why the below doesn't work
                # closestVertices = getClosestVerticesTopBase(vertices, baseObj.name, 2)

                # TODO: might get scaffolds where they're not connected like this..
                status = checkScaffoldConnectionBaseNoErr(baseObj.name, closestVertices)
                if status:
                    # generate edges using patch ids
                    edgesObjs.append([baseObj, foldpatchObj])
                    print("Adding edges to the list: ", baseObj.name, foldpatchObj.name)

        self.edgesObjs = edgesObjs

    def cleanUpSplitPatches(self):
        print("InputScaff: cleaning up split patches...")
        for bScaff in self.basicScaffoldWrappers:
            bScaff.cleanUpSplitPatches()

    def genInputScaffold(self):
        # Generate the input scaffold
        print("Generating Input Scaffold...")
        npAxis = np.array([self.pushAxis.x, self.pushAxis.y, self.pushAxis.z])
        edgesIds = [[e[0].id, e[1].id] for e in self.edgesObjs]

        # Create input scaffold object
        self.inputScaffold = fold.InputScaff(self.patchesObjs, edgesIds, npAxis, self.maxHinges, self.shrinks)

        # Generate hinge graph
        self.inputScaffold.gen_hinge_graph()

    def setDependencyScaffold(self, topPatchObj: fold.Patch, basePatchObj: fold.Patch, basicScaffWrapper: MayaHBasicScaffoldWrapper) -> bool:
        hasParentOrChild: bool = False
        for scaffWrapper in self.basicScaffoldWrappers:
            if topPatchObj.name == scaffWrapper.basePatch:
                # TODO: INVESTIGATE WHETHER THIS IS NECESSARY
                if (scaffWrapper.parent is None):
                    scaffWrapper.setParent(basicScaffWrapper)
                basicScaffWrapper.addChild(scaffWrapper)
                hasParentOrChild = True
                print("Parent child pairing found: parent: " + str(basicScaffWrapper.basicScaffold.id) + " child: " + str(scaffWrapper.basicScaffold.id))
            elif basePatchObj.name == scaffWrapper.patches[1]:
                if basicScaffWrapper.parent is None:
                    basicScaffWrapper.setParent(scaffWrapper)
                scaffWrapper.addChild(basicScaffWrapper)
                hasParentOrChild = True
                print("Parent child pairing ofund: parent: " + str(scaffWrapper.basicScaffold.id) + " child: " + str(
                        basicScaffWrapper.basicScaffold.id))
        return hasParentOrChild

    def genBasicScaffolds(self):

        print("Generating Basic Scaffolds...")
        if (self.inputScaffold == None):
            raise Exception("Input scaffold is not set yet!")

        if (len(self.edgesObjs) == 0):
            print("Error! No edges, yet genBasicScaffolds is called!")
            exit(1)

        # Create a dictionary where the key is the foldablePatch and the values are the patches it is connected to
        foldPatchObjDiction: Dict[fold.Patch, list[fold.Patch]] = {}
        for edgeObj in self.edgesObjs:
            if edgeObj[1] not in foldPatchObjDiction:
                foldPatchObjDiction[edgeObj[1]] = [edgeObj[0]]
            else:
                foldPatchObjDiction[edgeObj[1]].append(edgeObj[0])

        # For each entry in foldPatchDiction, create a basic scaffold
        for foldPatchObj in foldPatchObjDiction.keys():

            # print("Basic scaffold creation...")
            if len(foldPatchObjDiction[foldPatchObj]) == 2:

                # If the pushAxis is positive, then the basePatch with the lower value in that axis is basePatch
                # If the pushAxis is negative, then the basePatch with the higher value in that axis is basePatch
                basePatchObj0 = foldPatchObjDiction[foldPatchObj][0]
                basePatchObj1 = foldPatchObjDiction[foldPatchObj][1]

                ## should be a list of 3 items
                basePatch0Vertices = basePatchObj0.coords
                basePatch1Vertices = basePatchObj1.coords

                if self.pushAxis[1] > 0:
                    if basePatch0Vertices[0][1] < basePatch1Vertices[0][1]:
                        topPatchObj = basePatchObj0
                        basePatchObj = basePatchObj1
                    else:
                        topPatchObj = basePatchObj1
                        basePatchObj = basePatchObj0
                else:
                    if basePatch0Vertices[0][1] > basePatch1Vertices[0][1]:
                        topPatchObj = basePatchObj0
                        basePatchObj = basePatchObj1
                    else:
                        topPatchObj = basePatchObj1
                        basePatchObj = basePatchObj0

                basePatch = basePatchObj.name
                patchList = [foldPatchObj.name, topPatchObj.name]
                patchObjList = [basePatchObj, foldPatchObj, topPatchObj]

                # Create a basic scaffold with the foldPatch and the base patches it is connected to
                basicScaffWrapper = MayaHBasicScaffoldWrapper(patchObjList, basePatch, patchList, self.pushAxis, self.maxHinges, self.shrinks)

                self.setDependencyScaffold(topPatchObj, basePatchObj, basicScaffWrapper)

                self.addBasicScaffold(basicScaffWrapper)

                # TODO: DO THE SAME FOR T SCAFF UGH

        self.inputScaffold.set_basic_mappings()
        print("end gen basic scaffs")

    def genFoldSolutions(self):
        print("Genearating Fold Solutions...")
        # Responsible for invoking the foldMain APIs

        # generate connectivity info
        self.genConnectivityInfo()

        # generate the input scaffold
        self.genInputScaffold()

        # generate and insert the basic scaffs
        self.genBasicScaffolds()

        # generate the mid level scaffolds
        self.inputScaffold.gen_mid_scaffs()

        # generate solutions. After this step, each basic scaffold should have its favorite solution
        self.inputScaffold.fold()

    def foldAnimate(self, time, recreatePatches):
        # TODO: need to later figure out how to do this with mid level scaffolds first
        # TODO: assume list of basic scaffolds is not sorted in any way
        for bScaff in self.basicScaffoldWrappers:
            print("Folding basic scaffold: " + str(bScaff.basicScaffold.id))
            bScaff.foldAnimateBasic(time, recreatePatches)


# Node definition
class foldableNode(OpenMayaMPx.MPxNode):
    # Declare class variables:
    # TODO:: declare the input and output class variables
    #         i.e. inNumPoints = OpenMaya.MObject()

    # duration of movement
    inTime = OpenMaya.MObject()

    # number of hinges
    inNumHinges = OpenMaya.MObject()

    # maximum number of shrinks
    inNumShrinks = OpenMaya.MObject()

    inPatchList = OpenMaya.MObject()  # TODO make into inInitialpatches

    # Dummy output plug that can be connected to the input of an instancer node so our node can live somewhere
    outPoint = OpenMaya.MObject()

    # TODO: later on we will iterate through basicScaffolds instead
    defaultInputScaffWrapper: MayaInputScaffoldWrapper = None

    new_shapes = []

    prevNumHinges = -1
    prevShrinks = -1
    prevPushAxis = [-1, -1, -1]  # TODO: make more generic

    # TODO: would like to remove this in the future and not need it
    prevPatchList = []

    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

        # selected_objects = cmds.ls(selection=True)

        # print("Maya Node Constructor")
        # print("Selected objects: ")
        # print(selected_objects)
        #
        # for i, obj in enumerate(selected_objects):
        #     print("Selected object: " + obj)
        #     # Create a string attribute for each selected object
        #     cmds.addAttr(obj, longName='stringAttr', dataType='string')
        #
        #     # Set the value of the string attribute to the object's name
        #     cmds.setAttr(obj + '.stringAttr', obj, type='string')
        #
        #     # Create a child string attribute for the compound attribute in the custom node
        #     child_attr_name = 'patchListChild{}'.format(i)
        #     cmds.addAttr('foldableNode1', longName=child_attr_name, dataType='string', parent='patchList')
        #
        #     # Connect the string attribute to the custom node's patchList attribute
        #     cmds.connectAttr(obj + '.stringAttr', 'foldableNode1.' + child_attr_name)

        print("End Maya Node Constructor")


    # compute
    def compute(self, plug, data):
        # Print the MDGContext
        context = data.context().isNormal()

        if (context == False):
            print("Context is not normal, returning")
            return

        # TODO:: create the main functionality of the node. Your node should
        #         take in three floats for max position (X,Y,Z), three floats
        #         for min position (X,Y,Z), and the number of random points to
        #         be generated. Your node should output an MFnArrayAttrsData
        #         object containing the random points. Consult the homework
        #         sheet for how to deal with creating the MFnArrayAttrsData.

        timeData = data.inputValue(self.inTime)
        time = timeData.asInt()

        numHingeData = data.inputValue(self.inNumHinges)
        numHinges = numHingeData.asInt()

        numShrinksData = data.inputValue(self.inNumShrinks)  # TODO: Represents maximum allowed shrinks
        numShrinks = numShrinksData.asInt()

        stringListData = data.inputValue(self.inPatchList)
        stringList = stringListData.asString()

        # Parse the string list separated by commas into a list of individual strings
        patches = stringList.split(',')

        print("PATCHES:")
        print(patches)

        if (len(patches) == 0):
            print("No patches inputted")
            return

        # try:
        #     # Try to find an attribute called
        #     print("trying to get patches")
        #     patchesData = data.inputValue(self.inStringList)
        #     print("trying to get object...")
        #     patchesObject = patchesData.data()
        #     print("trying to cast to MFnStringArrayData...")
        #     if (patchesObject.isNull()):
        #         raise Exception("patchesObject is null")
        #
        #     patchesFn = OpenMaya.MFnStringArrayData(patchesObject)
        #     print("trying to get length...")
        #     patchListLength = patchesFn.length()
        #
        #     for i in range(patchListLength):
        #         stringElem = patchesFn[i]
        #         print(stringElem)
        #
        # except():
        #     print("No patches inputted")


        # TODO: Eventually remove, hard coded patches for now
        # patches = ["cBase", "cFold", "cFold1", "cTop", "cFold2", "cTop1", "cFold3", "cFold4", "cTop2"]
        # patches = ["pBaseBottomH", "pFoldH", "pBaseTopH"]
        # patches = ["mBase", "mFold1", "mFold2", "mTop", "mFold3", "mTop1"]
        # patches = ["dBase", "dFold1", "dFold2", "dTop"]
        # patches = ["gBase", "gFold1", "gFold2", "gBase1", "gFold3", "gBase2"]
        # patches = ["lBase", "lFold", "lBase1"]
        # patches = ["jBase1", "jFold1", "jBase2", "jFold2", "jBase3", "jFold3", "jBase4", "jFold4", "jBase5", "jFold5"]
        # patches = ["jBase1", "jBase5", "jFold1", "jBase4", "jBase2", "jFold2", "jBase3", "jFold4", "jFold5", "jFold3"]

        # patches = ["rBase1", "rFold1", "rFold2", "rFold3", "rBase2"]
        # patches = ["kBase1", "kFold1", "kBase2", "kFold2", "kBase3", "kFold3", "kBase4", "kFold4", "kBase5", "kFold5"]
        # patches = ["gtBase1", "gtFold1", "gtFold2", "gtBase2", "gtFold3", "gtBase3"]

        # b1Patches = ["mBase", "mFold1", "mTop"]
        # b2Patches = ["mBase", "mFold2", "mTop"]

        # TODO: hard code push axis for now
        pushAxis = [0, -1, 0]

        recreatePatches = False

        # If any of the input variables have changed, then create new scaffold
        if (self.prevNumHinges != numHinges or self.prevShrinks != numShrinks or self.prevPushAxis != pushAxis or
        self.prevPatchList != patches):

            # Reset variables
            self.prevNumHinges = numHinges
            self.prevShrinks = numShrinks
            self.prevPushAxis = pushAxis
            self.prevPatchList = patches

            # Current Scaffolds should clear their patches from the scene, if there is one
            if (self.defaultInputScaffWrapper != None):
                self.defaultInputScaffWrapper.cleanUpSplitPatches()

            # Reset the fold class's ids
            resetFoldClass()

            # Create new MayaInputScaffoldWrapper
            self.defaultInputScaffWrapper = MayaInputScaffoldWrapper(patches, OpenMaya.MVector(pushAxis[0], pushAxis[1],
                                                                                               pushAxis[2]), numHinges,
                                                                     numShrinks)

            # Run the fold algorithm
            self.defaultInputScaffWrapper.genFoldSolutions()

            recreatePatches = True

        # Regardless of whether we have a new scaffold or not, we need to animate the current one
        self.defaultInputScaffWrapper.foldAnimate(time, recreatePatches)

        # Mandatory data clean
        data.setClean(plug)


# initializer
def nodeInitializer():
    tAttr = OpenMaya.MFnTypedAttribute()
    nAttr = OpenMaya.MFnNumericAttribute()

    # TODO:: initialize the input and output attributes. Be sure to use the
    #         MAKE_INPUT and MAKE_OUTPUT functions

    try:
        print("Initialization!\n")
        foldableNode.inTime = nAttr.create("inTime", "t", OpenMaya.MFnNumericData.kInt, 0)
        MAKE_INPUT(nAttr)

        foldableNode.inNumHinges = nAttr.create("numHinges", "nH", OpenMaya.MFnNumericData.kInt, 3)
        MAKE_INPUT(nAttr)

        foldableNode.inNumShrinks = nAttr.create("numShrinks", "nS", OpenMaya.MFnNumericData.kInt, 2)
        MAKE_INPUT(nAttr)

        # print("CHECK POINT 1180")

        # foldableNode.inStringList = compoundFn.create('patchList', 'pl')

        # print("CHECK POINT 1188")

        # selected_objects = cmds.ls(selection=True)
        #
        # print("CHECK POINT 1192")
        #
        # for i, obj in enumerate(selected_objects):
        #     print("Selected object: " + obj)
        #     # Create a string attribute for each selected object
        #     cmds.addAttr(obj, longName='stringAttr', dataType='string')
        #
        #     # Set the value of the string attribute to the object's name
        #     cmds.setAttr(obj + '.stringAttr', obj, type='string')
        #
        #     # Create a child string attribute for the compound attribute in the custom node
        #     child_attr_name = 'patchListChild{}'.format(i)
        #     cmds.addAttr('foldableNode1', longName=child_attr_name, dataType='string', parent='patchList')
        #
        #     # Connect the string attribute to the custom node's patchList attribute
        #     cmds.connectAttr(obj + '.stringAttr', 'foldableNode1.' + child_attr_name)

        # print("CHECK POINT 1210")

        # defaultList = OpenMaya.MFnStringArrayData().create()
        # foldableNode.inStringList = tAttr.create("patchList", "pL", OpenMaya.MFnStringArrayData.kStringArray)
        # MAKE_INPUT(tAttr)

        defaultList = OpenMaya.MFnStringData().create("")
        foldableNode.inPatchList = tAttr.create("patchList", "pL", OpenMaya.MFnData.kString, defaultList)
        MAKE_INPUT(tAttr)

        foldableNode.textList = tAttr.create("textList", "tL", OpenMaya.MFnStringArrayData.kStringArray)
        MAKE_INPUT(tAttr)

        foldableNode.outPoint = tAttr.create("outPoint", "oP", OpenMaya.MFnArrayAttrsData.kDynArrayAttrs)
        MAKE_OUTPUT(tAttr)

    except Exception as e:
        print(e)
        sys.stderr.write(("Failed to create attributes of %s node\n", kPluginNodeTypeName))

    try:
        # TODO:: add the attributes to the node and set up the
        #         attributeAffects (addAttribute, and attributeAffects)
        foldableNode.addAttribute(foldableNode.inTime)
        foldableNode.addAttribute(foldableNode.inNumHinges)
        foldableNode.addAttribute(foldableNode.inNumShrinks)
        foldableNode.addAttribute(foldableNode.inPatchList)
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inTime, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumHinges, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumShrinks, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inPatchList, foldableNode.outPoint)

    except Exception as e:
        print(e)
        sys.stderr.write(("Failed to add attributes of %s node\n", kPluginNodeTypeName))


# creator
def nodeCreator():
    return OpenMayaMPx.asMPxPtr(foldableNode())


# initialize the script plug-in
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.registerNode(kPluginNodeTypeName, foldableNodeId, nodeCreator, nodeInitializer)
    except:
        sys.stderr.write("Failed to register node: %s\n" % kPluginNodeTypeName)

    # Load menu
    print("Executing Command...\n")
    OpenMaya.MGlobal.executeCommand("source \"" + mplugin.loadPath() + "/unhingedDialogue.mel\"")


# uninitialize the script plug-in
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)

    OpenMaya.MGlobal.executeCommand("source \"" + mplugin.loadPath() + "/removeMenu.mel\"")

    try:
        mplugin.deregisterNode(foldableNodeId)
    except:
        sys.stderr.write("Failed to unregister node: %s\n" % kPluginNodeTypeName)
