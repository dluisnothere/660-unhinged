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


# Static helper functions
def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)
    return OpenMaya.MFnTransform(transform_dag_path)


def getObjectObjectFromDag(name: str) -> OpenMaya.MDagPath:
    # Create an MSelectionList object to store the plane name
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


def checkScaffoldConnection(pivot: OpenMaya.MVector, middlepoint: OpenMaya.MVector):
    dist = OpenMaya.MVector(pivot - middlepoint).length()
    print("Pivot distance to middle point: {:.6f}".format(dist))
    if dist > 0.0001:
        print("Error: Distance is not 0. Patches are not connected")
        exit(1)


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
        print("vertex: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))
        if abs(vertex[1] - childY) > 0.0001:
            print("Y values are not the same!")
            print("Parent Y: {}".format(vertex[1]))
            print("Child Y: {}".format(childY))
            connected = False
            break
        if vertex[0] < childMinX:
            print("X value is less than minX")
            connected = False
            break
        if vertex[0] > childMaxX:
            print("X value is larger than maxX")
            connected = False
            break
        if vertex[2] < childMinZ:
            print("Z value is smaller childMinZ")
            connected = False
            break
        if vertex[2] > childMaxZ:
            print("Z value is larger than childMaxZ")
            connected = False
            break

    if not connected:
        print("Error: Patches are not connected")
        exit(1)


# def checkScaffoldConnectionNoErr(pivot: OpenMaya.MVector, middlepoint: OpenMaya.MVector) -> bool:
#     dist = OpenMaya.MVector(pivot - middlepoint).length()
#     print("Pivot distance to middle point: {:.6f}".format(dist))
#     if dist > 0.0001:
#         return False
#     return True

def checkScaffoldConnectionBaseNoErr(base: str, foldable) -> bool:
    print("CHECKING SCAFFOLD CONNECTION BASE...")
    print("base: {}".format(base))
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
        print("verices in foldable closest to base: {:.6f}, {:.6f}, {:.6f}".format(vertex[0], vertex[1], vertex[2]))
        if abs(vertex[1] - baseY) > 0.0001:
            print("Y values are not the same!")
            print("Parent Y: {}".format(vertex[1]))
            print("Child Y: {}".format(baseY))
            connected = False
            break
        if vertex[0] < baseMinX:
            print("X value is less than minX")
            connected = False
            break
        if vertex[0] > baseMaxX:
            print("X value is larger than maxX")
            connected = False
            break
        if vertex[2] < baseMinZ:
            print("Z value is smaller childMinZ")
            connected = False
            break
        if vertex[2] > baseMaxZ:
            print("Z value is larger than childMaxZ")
            connected = False
            break

    return connected


def isPolyPlane(obj):
    # Create an MSelectionList object
    transformDagPath = getObjectObjectFromDag(obj)

    # Get the shape node
    print("Getting shape node")
    transformDagPath.extendToShape()

    print("Checking if shape node is of type mesh")
    # Check if the shape node is of type "mesh"
    if transformDagPath.node().hasFn(OpenMaya.MFn.kMesh):
        print("creating mesh")
        # Create an MFnMesh function set
        fnMesh = OpenMaya.MFnMesh(transformDagPath)

        # Get the number of faces in the mesh
        numFaces = fnMesh.numPolygons()

        # If the mesh has only one face, it can be considered a polygonal plane
        if numFaces == 1:
            return True
    return False


class MayaHBasicScaffoldWrapper():
    inInitialPatches = []

    # The non base patches of the H scaff
    shapeTraverseOrder: List[str] = []
    shapeBase = []
    shapeResetTransforms = {}

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
    basicScaffold = None

    # TODO: for now, use FoldManager instead of basicScaffold to output hard coded results
    foldManager = None
    foldManagerOption = None

    def __init__(self, basePatch: str, patches: List[str], pushAxis: OpenMaya.MVector, maxHinges: int, shrinks: int):
        self.basePatch = basePatch
        self.patches = patches
        self.shapeTraverseOrder = []

        self.pushAxis = pushAxis

        self.maxHinges = maxHinges
        self.shrinks = shrinks

        self.newShapes = []


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
        for i in range(0, len(self.newShapes)):
            cmds.delete(self.newShapes[i])

        # Clear the new shapes list.
        self.newShapes = []

    def setUpGenericScene(self, upperPatches: List[str], basePatch: str):
        print("Setting up Generic scene...")
        # TODO: theoretically we should only need to move things in the upper patches
        # Get the transforms for each item in upper patches
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
        print("Restoring Initial State...")
        self.shapeTraverseOrder = self.getPatches()
        print("restored Patches: ")
        print(self.shapeTraverseOrder)

        # Clears self.shapeResetTransforms
        self.shapeResetTransforms = {}

        # fill new_translation and new_rotation with original values
        for shape in self.shapeTraverseOrder:
            transform = getObjectTransformFromDag(shape)
            translate = transform.translation(OpenMaya.MSpace.kWorld)
            rotate = cmds.getAttr(shape + ".rotate")

            # Reset self.shapeResetTransforms
            self.shapeResetTransforms[shape] = [translate, rotate]

        # Reset back to original shapes so you can break them again
        self.cleanUpSplitPatches()

    def shrinkPatch(self, shapeTraverseOrder, endPiece, numPieces, startPiece):
        print("shrinking patches...")
        for i in range(0, len(shapeTraverseOrder) - 1):
            foldable_patch = shapeTraverseOrder[i]
            middle = 1 / 2  # hard coded for now

            # Translate patch to the new midpoint
            pieceWidth = 1.0 / numPieces  # hard coded for now
            newMiddle = (startPiece + endPiece) * pieceWidth / 2

            print("newMiddle in Z direction: {}".format(newMiddle))
            transform = getObjectTransformFromDag(foldable_patch)
            translation = OpenMaya.MVector(0, 0, newMiddle) - OpenMaya.MVector(0, 0, middle)
            print("translation: {}".format(translation))
            transform.translateBy(translation, OpenMaya.MSpace.kTransform)

            # Shrink patch by numPieces in the hard coded z direction
            transform = getObjectTransformFromDag(foldable_patch)
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
        # TODO: make this more generic, currently assumes foldable patch is at the center
        foldablePatches = ["pFoldH"]
        for patch in foldablePatches:
            cmds.setAttr(patch + ".visibility", False)

        print("break patches called")
        for j in range(0, len(shapeTraverseOrder) - 1):  # every patch except last guy is foldable
            foldablePatch = shapeTraverseOrder[
                j]  # TODO: make more generic, currently assumes foldable patch is at the center

            print("debugging for existence: breakPatches: 447 ALREADY DOESN'T EXIST")
            debugOrigScaleX = cmds.getAttr(foldablePatch + ".scaleX")
            print("debugOriginalScaleX: " + str(debugOrigScaleX))

            shapeTraverseOrder.remove(foldablePatch)
            del self.shapeResetTransforms[foldablePatch]

            print("debugging for existence: breakPatches: 454 ALREADY DOESN'T EXIST")
            debugOrigScaleX = cmds.getAttr(foldablePatch + ".scaleX")
            print("debugOriginalScaleX: " + str(debugOrigScaleX))

            newPatches, newTransforms = self.generateNewPatches(foldablePatch, numHinges)

            # Add the new patch transforms to the shape_reset_transforms and insert new patches to shape_traverse_order
            for i in range(0, len(newPatches)):
                shapeTraverseOrder.insert(j, newPatches[i])
                self.shapeResetTransforms[newPatches[i]] = [newTransforms[i][0], newTransforms[i][1]]

                # Keep track of the new patches just created so we can delete it on the next iteration
                self.newShapes.append(newPatches[i])

    def getPatchPivots(self, shapeTraverseOrder: List[str]) -> List[OpenMaya.MPoint]:
        patchPivots = []
        for shape in shapeTraverseOrder:
            pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
            patchPivots.append(pivot)
            print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))
        return patchPivots

    def findClosestMidpointsOnPatches(self, patchPivots: List[OpenMaya.MPoint], shapeTraverseOrder: List[str]) -> (
            List[List], List[float]):
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
            currentClosest = getClosestVertices(bottomVertices, childPivot, 2)
            closestVertices.append(currentClosest)
            print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(closestVertices[vertId][0][0],
                                                                                      closestVertices[vertId][0][1],
                                                                                      closestVertices[vertId][0][2][0],
                                                                                      closestVertices[vertId][0][2][1],
                                                                                      closestVertices[vertId][0][2][2]))
            print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(closestVertices[vertId][1][0],
                                                                                      closestVertices[vertId][1][1],
                                                                                      closestVertices[vertId][1][2][0],
                                                                                      closestVertices[vertId][1][2][1],
                                                                                      closestVertices[vertId][1][2][2]))

            # Get the middle point between the two vertices.
            verticeDist = closestVertices[vertId][0][2] + closestVertices[vertId][1][2]
            print("Vertice Dist: {:.6f}, {:.6f}, {:.6f}".format(verticeDist[0], verticeDist[1], verticeDist[2]))
            middlePoint = (verticeDist * 0.5)
            print("Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePoint[0], middlePoint[1], middlePoint[2]))

            midPoints.append(middlePoint)

            # Ensure the parent and child are actually connected
            # TODO: generalize to T scaffolds as well
            if (i == len(shapeTraverseOrder) - 2):
                checkScaffoldConnectionTopBase(currentClosest, child)
            else:
                checkScaffoldConnection(childPivot, middlePoint)

        return closestVertices, midPoints

    # TODO: might be a member function of basic scaff
    def rotatePatches(self, angle: float, rotAxis: List[float], shapeTraverseOrder: List[str], isLeft: bool) -> List[
        OpenMaya.MFnTransform]:
        patchTransforms = []
        if (isLeft):
            angle = -angle

        for i in range(0, len(shapeTraverseOrder)):
            shape = shapeTraverseOrder[i]
            pTransform = getObjectTransformFromDag(shape)
            patchTransforms.append(pTransform)
            if (i == len(shapeTraverseOrder) - 1):  # TODO: fix this bc it won't work for T scaffolds
                break

            q = OpenMaya.MQuaternion(math.radians(angle), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
            print("angle" + str(angle))
            print("q: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(q[0], q[1], q[2], q[3]))
            pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)

            angle = -angle
        return patchTransforms

    def updatePatchTranslations(self, closestVertices: List, midPoints: List, patchPivots: List, patchTransforms: List,
                                shapeTraverseOrder: List[str]):
        # Get the new closest vertices without changing the original closest vertices
        newClosestVertices = closestVertices.copy()
        for i in range(0, len(patchPivots) - 1):
            # Obtain child pivot so we can use it later for translation
            childPivot = patchPivots[i + 1]
            for j in range(0, len(newClosestVertices[
                                      i])):  # index and use information from updated vertex positions. There should only be 2 verts here
                vertex_name, dist, vertexPoint = newClosestVertices[i][j]

                # Get the world position of the vertex and convert it to an MVector
                vertexPoint = cmds.pointPosition(vertex_name, world=True)
                vertexPoint = OpenMaya.MVector(vertexPoint[0], vertexPoint[1], vertexPoint[2])

                # Translate pivot to the closest vertex's mid point,b ut we don't want that
                # dist = OpenMaya.MVector(childPivot - vertexPoint).length()

                newClosestVertices[i][j] = (
                    vertex_name, 0,  # not sure if dist is important anymore
                    vertexPoint)

                # Print new location and distance.
                print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(newClosestVertices[i][j][0],
                                                                                          newClosestVertices[i][j][1],
                                                                                          newClosestVertices[i][j][2][
                                                                                              0],
                                                                                          newClosestVertices[i][j][2][
                                                                                              1],
                                                                                          newClosestVertices[i][j][2][
                                                                                              2]))

            # Midpoint formula to solve for the midpoint betwen the two closest vertices.
            verticeDistNew = newClosestVertices[i][0][2] + newClosestVertices[i][1][2]
            middlePointNew = (verticeDistNew * 0.5)
            print(
                "Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePointNew[0], middlePointNew[1], middlePointNew[2]))

            # Get the translation from the old middle point to the new middle point.
            ogMidPoint = midPoints[i]
            translation = middlePointNew - ogMidPoint
            print("Middle point translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1],
                                                                            translation[2]))

            # Translate child patch by the translation.
            print("Translating child patch: " + shapeTraverseOrder[i + 1])
            childPatchTransform = patchTransforms[i + 1]
            print("Translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1], translation[2]))
            childPatchTransform.translateBy(translation, OpenMaya.MSpace.kWorld)

    def genBestFoldOption(self):
        # Return the hard coded object from foldManager for now
        patchVertices = self.getAllPatchVertices()
        patchVertices = np.array(patchVertices)

        self.foldManager = fold.FoldManager()
        self.foldManager.generate_h_basic_scaff(patchVertices[0], patchVertices[1], patchVertices[2])

        self.foldManagerOption = self.foldManager.mainFold(self.maxHinges, self.shrinks)

    # Splits the foldTest function into two parts.
    def foldKeyframe(self, time, shapeTraverseOrder: List[str], foldSolution: fold.FoldOption, recreatePatches: bool):
        # Get the relevant information from the fold_solution
        startAngles = foldSolution.fold_transform.startAngles
        endAngles = foldSolution.fold_transform.endAngles

        isLeft = foldSolution.isleft

        # Hinge variables
        numHinges = foldSolution.modification.num_hinges

        # Shrinking variables
        startPiece = foldSolution.modification.range_start
        endPiece = foldSolution.modification.range_end  # non inclusive
        numPieces = foldSolution.modification.num_pieces

        t = time  # dictate that the end time is 90 frames hard coded for now

        # TODO: let author dictate end time.. but this doesn't realy matter.
        angle = t * (endAngles[0] - startAngles[0]) / 90  # The angle we fold at this particular time is time / 90 *
        # TODO: make more generic in the future
        rotAxis = (0, 0, 1)

        print("angle based on t: " + str(angle))
        print("t: " + str(t))

        # print("debugging for existence: foldKeyframe, assume it's in index 0: ALREADY DOESN'T EXIST")
        # print("shapeTraverseOrder: " + str(shapeTraverseOrder))
        # debugOrigScaleX = cmds.getAttr(shapeTraverseOrder[0] + ".scaleX")
        # print("debugOriginalScaleX: " + str(debugOrigScaleX))

        # Update the list of shape_traverse_order to include the new patches where the old patch was
        if (recreatePatches and numHinges > 0):
            self.breakPatches(shapeTraverseOrder, numHinges)

        # Moved from the recreate_patches condition because we always want this to be visible if no hinges
        # TODO: ventually move this to a better place
        if (numHinges == 0):
            # set middle patch to be visible
            cmds.setAttr(shapeTraverseOrder[0] + ".visibility", 1)

        # Loop through the patches and get all of their pivots.
        patchPivots = self.getPatchPivots(shapeTraverseOrder)

        # Find the closest vertices to the patch pivots and calculate the midPoints, also check scaff is connected
        closestVertices, midPoints = self.findClosestMidpointsOnPatches(patchPivots, shapeTraverseOrder)

        # Perform rotations at once, but do not rotate the last patch
        patchTransforms = self.rotatePatches(angle, rotAxis, shapeTraverseOrder, isLeft)

        # Update location of closest vertices after rotation and update children translations
        self.updatePatchTranslations(closestVertices, midPoints, patchPivots, patchTransforms, shapeTraverseOrder)

        # Has to go at the end or something otherwise you'll get a space between top patch and the folds
        self.shrinkPatch(shapeTraverseOrder, endPiece, numPieces, startPiece)

    # Fold test for non hard coded transforms: Part 1 of the logic from foldTest, calls foldKeyframe()
    # AT this point should already have the best fold option
    def foldGeneric(self, time: int, recreatePatches: bool):
        # foldOption = self.basicScaffold.fold_options[0]
        foldOption = self.foldManagerOption
        startTime = foldOption.fold_transform.startTime
        endTime = foldOption.fold_transform.endTime

        # If folding for this scaff hasn't started yet, don't do anything
        if (time >= startTime):
            if (time >= endTime):
                # Cap animation at endTime
                # TODO: might make it so that it doesn't even translate after endTime but not sure.
                time = endTime

            # if (len(self.shapeTraverseOrder) != 0):
            #     print("debugging for existence: foldGeneric 675, assume it's in index 0")
            #     print("shapeTraverseOrder: " + str(self.shapeTraverseOrder))
            #     debugOrigScaleX = cmds.getAttr(self.shapeTraverseOrder[0] + ".scaleX")
            #     print("debugOriginalScaleX: " + str(debugOrigScaleX))

            print("BasicScaffold's patches during foldGeneric: CHANGED ALREALDY")
            print(self.getPatches())

            self.inInitialPatches = self.getPatchesIncludeBase()

            # For now we create an input scaffold with allPatches and call genConnectivityInfo on it
            # self.defaultInputScaff = MayaInputScaffold(allPatches, OpenMaya.MVector(pushAxis[0], pushAxis[1], pushAxis[2]))
            # self.defaultInputScaff.genConnectivityInfo()

            # self.defaultScaff = MayaHBasicScaffold(self.inInitialPatches[0], self.inInitialPatches[1:])
            # self.basicScaffolds.append(self.defaultScaff)

            # If self.shape_traverse_order is empty, we fill it with original shapes
            # No need to reset the scene if it hasn't been changed yet.

            # recreatePatches = (self.num_hinges != numHinges)
            # if (len(self.shapeTraverseOrder) != 0):
            #     print("debugging for existence: foldGeneric, assume it's in index 0: EXISTS HERE.")
            #     print("shapeTraverseOrder: " + str(self.shapeTraverseOrder))
            #     debugOrigScaleX = cmds.getAttr(self.shapeTraverseOrder[0] + ".scaleX")
            #     print("debugOriginalScaleX: " + str(debugOrigScaleX))

            if (len(self.shapeTraverseOrder) == 0 or recreatePatches):
                # self.num_hinges = numHinges
                self.restoreInitialState()

            else:
                # Reset the scene
                # TODO; Might not work anymore in a bit
                self.setUpGenericScene(self.shapeTraverseOrder, self.shapeBase)

            # # Get the vertices of each patch in the list and create a FoldManager using it.
            # patchVerticesList = self.defaultScaff.getAllPatchVertices()
            # patchVerticesList = np.array(patchVerticesList)  # TODO: eventaully might not need this...
            # manager = fold.FoldManager()
            # manager.generate_h_basic_scaff(patchVerticesList[0], patchVerticesList[1], patchVerticesList[2])

            # Generate the solution from the foldManager
            # solution = manager.mainFold(numHinges, numShrinks)
            # solution = self.foldManagerOption

            # Call the keyframe funtion
            # TODO: get rid of these params since some of them are just member vars
            self.foldKeyframe(time, self.shapeTraverseOrder, foldOption, recreatePatches)


class MayaInputScaffoldWrapper():
    def __init__(self, patches: List[str], pushAxis: OpenMaya.MVector, nH: int, nS: int):
        self.pushAxis = pushAxis
        self.patches = patches
        self.bases = []
        self.foldables = []
        self.edges = []
        self.maxHinges = nH
        self.shrinks = nS

        self.inputScaffold = None
        self.basicScaffolds: List[MayaHBasicScaffoldWrapper] = []

        print("Length of basicScaffolds on init: " + str(len(self.basicScaffolds)))

    def getPatches(self) -> List[str]:
        return self.patches

    def genConnectivityInfo(self):
        # Test each patch for connectivity to other patches
        # First, only get the patches that are normal to the pushing direction called base patch
        # For each base patch, test for connectivity against all other patches (foldable patches)
        # If they are close enough to each other via check-scaffold connectivity, then add an edge between them in the form of
        # [base_patch, foldable_patch]

        for patch in self.patches:
            # Get the surface normal of the patch in world space
            print("getting surface normal for {}".format(patch))
            planeDagPath = getObjectObjectFromDag(patch)
            fnMesh = OpenMaya.MFnMesh(planeDagPath)

            # Get the normal of the plane's first face (face index 0)
            # Note: If the plane has multiple faces, specify the desired face index
            normal = OpenMaya.MVector()
            # Apparently the normal agument is the SECOND argument in this dumbass function
            fnMesh.getPolygonNormal(0, normal, OpenMaya.MSpace.kWorld)

            print("normal: {:.6f}, {:.6f}, {:.6f}".format(normal[0],
                                                          normal[1],
                                                          normal[2]))

            # Get the dot product of normal and pushDir
            dot = self.pushAxis * normal
            if (abs(abs(dot) - 1) < 0.0001):
                # Parallel
                self.bases.append(patch)
            else:
                self.foldables.append(patch)

            print("basePatches")
            print(self.bases)

            print("foldablePatches")
            print(self.foldables)

        edges = []

        # For every base in basePatches, test for connectivity with every foldable_patch
        for base in self.bases:
            for foldpatch in self.foldables:
                # Since this is at the very beginning, checkScaffoldConnection should work as is
                # Find pivot of base
                pivot = getObjectTransformFromDag(base).rotatePivot(OpenMaya.MSpace.kWorld)

                # find the closest vertices from fold to pivot
                vertices = getObjectVerticeNamesAndPositions(foldpatch)
                closestVertices = getClosestVertices(vertices, pivot, 2)

                # Check if the middle point is close enough to the pivot
                middlePoint = (closestVertices[0][2] + closestVertices[1][2]) / 2

                # TODO: might get scaffolds where they're not connected like this...
                status = checkScaffoldConnectionBaseNoErr(base, closestVertices)
                # status = checkScaffoldConnectionNoErr(pivot, middlePoint)
                if status:
                    edges.append([base, foldpatch])

        print("Edges:")
        print(edges)

        self.edges = edges

    def genInputScaffold(self):
        # TODO: not super important yet
        print("genInputScaffold: Implement me!")

    def genFoldSolutions(self):
        # TODO: not super important yet
        # Call genFoldSolution of the input scaffold, which will hopefully populate each basic scaffold with a solution and the time interval of the solution
        print("genFoldSolutions...")
        for bScaff in self.basicScaffolds:
            bScaff.genBestFoldOption()

    def fold(self, time, recreatePatches):
        # Given that we have each basic scaffold with a solution, take in the current time and see the fold status of each basic scaffold.
        # For each basic scaffold, if the solution's startTime is less than the current time, then fold it with some animations
        # TODO: need to later figure out how to do this with mid level scaffolds first
        # TODO: assume list of basic scaffolds is not sorted in any way
        print("folding...")
        for bScaff in self.basicScaffolds:
            bScaff.foldGeneric(time, recreatePatches)


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

    # inStringList = OpenMaya.MObject()  # TODO make into inInitialpatches

    # Dummy output plug that can be connected to the input of an instancer node
    # so our node can "live" somewhere.
    outPoint = OpenMaya.MObject()

    # basicScaffolds: List[MayaHBasicScaffold] = []

    # TODO: later on we will iterate through basicScaffolds instead
    # defaultScaff: MayaHBasicScaffold = None
    defaultInputScaffWrapper: MayaInputScaffoldWrapper = None

    # shapeTraverseOrder: List[str] = []
    # shapeBase = []
    # shapeResetTransforms = {}

    new_shapes = []

    prevNumHinges = -1
    prevShrinks = -1
    prevPushAxis = [-1, -1, -1] # TODO: make more generic


    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

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

        # hardcode patches for now for an input scaffold which contains some mid level scaffolds in arbitrary order
        # patches = ["pBaseBottomMid", "pFoldMid", "pBaseTopMid", "pFoldMid2", "pBaseTopMid2", "pFoldMid3",
        #            "pBaseTopMid3", "pFoldMid4"]
        patches = ["pBaseBottomH", "pFoldH", "pBaseTopH"]

        # hard code push axis
        pushAxis = [0, -1, 0]

        recreatePatches = False
        # TODO: maybe only need to create a new MayaInputScaffoldWrapper if patches, pushAxis, numHinges, numShrinks has changed
        if (self.prevNumHinges != numHinges or self.prevShrinks != numShrinks or self.prevPushAxis != pushAxis):
            # Reset variables
            self.prevNumHinges = numHinges
            self.prevShrinks = numShrinks
            self.prevPushAxis = pushAxis

            # Create new MayaInputScaffoldWrapper
            self.defaultInputScaffWrapper = None
            self.defaultInputScaffWrapper = MayaInputScaffoldWrapper(patches, OpenMaya.MVector(pushAxis[0], pushAxis[1], pushAxis[2]), numHinges, numShrinks)

            print("NUMBER OF BASIC SCAFFS IN INPUT SCAFF 923: why is it already 1?? SUS POINT")
            print(len(self.defaultInputScaffWrapper.basicScaffolds))

            self.defaultInputScaffWrapper.genConnectivityInfo()
            self.defaultInputScaffWrapper.genInputScaffold()

            print("InputScaff's patches right after creation")
            print(self.defaultInputScaffWrapper.getPatches())

            # For now just hard code the basic scaffold in the input scaffold
            basicScaff = MayaHBasicScaffoldWrapper(patches[0], patches[1:], OpenMaya.MVector(pushAxis[0], pushAxis[1], pushAxis[2]), numHinges, numShrinks)

            print("BasicScaff's patches right after creation")
            print(basicScaff.getPatches())

            print("NUMBER OF BASIC SCAFFS IN INPUT SCAFF 938: why is it already 1?? SUS POINT")
            print(len(self.defaultInputScaffWrapper.basicScaffolds))

            self.defaultInputScaffWrapper.basicScaffolds.append(basicScaff) # this appends a copy of basicScaff

            print("BasicScaff's patches right after appending to input scaff")
            print(self.defaultInputScaffWrapper.basicScaffolds[0].getPatches())

            self.defaultInputScaffWrapper.genFoldSolutions()

            print("BasicScaff's patches right after GenFoldSolutions: ALREADY BAD")
            print(self.defaultInputScaffWrapper.basicScaffolds[0].getPatches())

            recreatePatches = True


        # always perform this step regardless
        self.defaultInputScaffWrapper.fold(time, recreatePatches)

        # self.foldTest(time)
        # self.foldGeneric(patches, pushAxis, time, numHinges, numShrinks)

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

        foldableNode.inNumShrinks = nAttr.create("numShrinks", "nS", OpenMaya.MFnNumericData.kInt, 4)
        MAKE_INPUT(nAttr)

        # defaultList = OpenMaya.MFnStringArrayData().create()
        # foldableNode.inStringList = tAttr.create("initialPatches", "iP", OpenMaya.MFnStringArrayData.kStringArray,
        #                                          defaultList)
        # MAKE_INPUT(tAttr)

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
        # foldableNode.addAttribute(foldableNode.inStringList)
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inTime, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumHinges, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumShrinks, foldableNode.outPoint)
        # foldableNode.attributeAffects(foldableNode.inStringList, foldableNode.outPoint)

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
