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
foldableNodeId = OpenMaya.MTypeId(0x8708)


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

def checkScaffoldConnectionNoErr(pivot: OpenMaya.MVector, middlepoint: OpenMaya.MVector) -> bool:
    dist = OpenMaya.MVector(pivot - middlepoint).length()
    print("Pivot distance to middle point: {:.6f}".format(dist))
    if dist > 0.0001:
        return False
    return True
def getPatchConnectivity(patches: List[str], pushDir: OpenMaya.MVector) -> List[List[str]]:
    # Test each patch for connectivity to other patches
    # First, only get the patches that are normal to the pushing direction called base patch
    # For each base patch, test for connectivity against all other patches (foldable patches)
    # If they are close enough to each other via check-scaffold connectivity, then add an edge between them in the form of
    # [base_patch, foldable_patch]

    basePatches = []
    foldablePatches = []
    for patch in patches:
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
        dot = pushDir * normal
        if (abs(abs(dot) - 1) < 0.0001):
            # Parallel
            basePatches.append(patch)
        else:
            foldablePatches.append(patch)

        print("basePatches")
        print(basePatches)

        print("foldablePatches")
        print(foldablePatches)

    edges = []

    # For every base in basePatches, test for connectivity with every foldable_patch
    for base in basePatches:
        for foldpatch in foldablePatches:
            # Since this is at the very beginning, checkScaffoldConnection should work as is
            # Find pivot of base
            pivot = getObjectTransformFromDag(base).rotatePivot(OpenMaya.MSpace.kWorld)

            # find the closest vertices from fold to pivot
            vertices = getObjectVerticeNamesAndPositions(foldpatch)
            closestVertices = getClosestVertices(vertices, pivot, 2)

            # Find the middle point of the closest vertices
            middlePoint = (closestVertices[0][2] + closestVertices[1][2]) / 2

            # Check if the middle point is close enough to the pivot
            # TODO: might get scaffolds where they're not connected like this...
            status = checkScaffoldConnectionNoErr(pivot, middlePoint)
            if status:
                edges.append([base, foldpatch])

    print("Edges:")
    print(edges)

    return edges

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

class MayaHBasicScaffold():
    def __init__(self, basePatch: str, patches: List[str]):
        self.basePatch = basePatch
        self.patches = patches

    def getBasePatch(self) -> str:
        return self.basePatch

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


class InputScaffold():
    def __init__(self, patches: List[str], pushDir: OpenMaya.MVector):
        self.pushDir = pushDir
        self.patches = patches
        self.bases = []
        self.foldables = []
        self.edges = []

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
            dot = self.pushDir * normal
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

                # Find the middle point of the closest vertices
                middlePoint = (closestVertices[0][2] + closestVertices[1][2]) / 2

                # Check if the middle point is close enough to the pivot
                # TODO: might get scaffolds where they're not connected like this...
                status = checkScaffoldConnectionNoErr(pivot, middlePoint)
                if status:
                    edges.append([base, foldpatch])

        print("Edges:")
        print(edges)

        self.edges = edges

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

    # TODO: make an OpenMaya.MObject() eventually
    inInitialPatches = []

    # Dummy output plug that can be connected to the input of an instancer node
    # so our node can "live" somewhere.
    outPoint = OpenMaya.MObject()

    basicScaffolds: List[MayaHBasicScaffold] = []

    # TODO: later on we will iterate through basicScaffolds instead
    defaultScaff: MayaHBasicScaffold = None

    shapeTraverseOrder: List[str] = []
    shapeBase = []
    shapeResetTransforms = {}

    new_shapes = []

    prev_num_hinges = 0

    num_hinges = 0

    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

    def cleanUpSplitPatches(self):
        for i in range(0, len(self.new_shapes)):
            cmds.delete(self.new_shapes[i])

        # Clear the new shapes list.
        self.new_shapes = []

    def setUpGenericScene(self, upperPatches: List[str], basePatch: str):
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
        self.shapeTraverseOrder = self.defaultScaff.getPatches()

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
        new_transforms = []
        for i in range(0, len(newPatches)):
            newTranslate = [originalTranslation[0][0], originalPatchBottom + newPatchScale * (i + 0.5),
                            originalTranslation[0][2]]
            newPatchPositions.append(newTranslate)
            cmds.setAttr(newPatches[i] + ".translate", newTranslate[0], newTranslate[1], newTranslate[2])

            # Append new patch transform to list of new transforms
            # Which will be used for its scene reset at the beginning
            # TODO: why is this a tuple with the name in the first spot?
            new_transforms.append([newTranslate, originalRotation])

        # Pivot the patches.
        for i in range(0, len(newPatches)):
            # Set the pivot location to the bottom of the patch
            newPivot = [newPatchScale * 0.5, 0, 0]
            transform = getObjectTransformFromDag(newPatches[i])
            transform.setRotatePivot(OpenMaya.MPoint(newPivot[0], newPivot[1], newPivot[2]), OpenMaya.MSpace.kTransform,
                                     True)

        newPatches.reverse()
        new_transforms.reverse()

        return newPatches, new_transforms

    def breakPatches(self, shapeTraverseOrder: List[str], numHinges: int):

        # Render the original foldable patch invisible
        foldablePatches = ["pFoldH"]
        for patch in foldablePatches:
            cmds.setAttr(patch + ".visibility", False)

        for j in range(0, len(shapeTraverseOrder) - 1):  # every patch except last guy is foldable
            foldablePatch = shapeTraverseOrder[
                j]  # TODO: make more generic, currently assumes foldable patch is at the center
            shapeTraverseOrder.remove(foldablePatch)
            del self.shapeResetTransforms[foldablePatch]

            newPatches, newTransforms = self.generateNewPatches(foldablePatch, numHinges)

            # Add the new patch transforms to the shape_reset_transforms and insert new patches to shape_traverse_order
            for i in range(0, len(newPatches)):
                shapeTraverseOrder.insert(j, newPatches[i])
                self.shapeResetTransforms[newPatches[i]] = [newTransforms[i][0], newTransforms[i][1]]

                # Keep track of the new patches just created so we can delete it on the next iteration
                self.new_shapes.append(newPatches[i])

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
    def foldGeneric(self, time, numHinges, numShrinks):

        # TODO: Should be input by the author
        self.inInitialPatches = ["pBaseBottomH", "pFoldH", "pBaseTopH"]

        # For now, we manually decompose inInitialPatches
        self.defaultScaff = MayaHBasicScaffold(self.inInitialPatches[0], self.inInitialPatches[1:])
        self.basicScaffolds.append(self.defaultScaff)

        # If self.shape_traverse_order is empty, we fill it with original shapes
        # No need to reset the scene if it hasn't been changed yet.
        recreatePatches = (self.num_hinges != numHinges)
        if (len(self.shapeTraverseOrder) == 0 or recreatePatches):
            self.num_hinges = numHinges
            self.restoreInitialState()

        else:
            # Reset the scene
            self.setUpGenericScene(self.shapeTraverseOrder, self.shapeBase)

        # Get the vertices of each patch in the list and create a FoldManager using it.
        patchVerticesList = self.defaultScaff.getAllPatchVertices()
        patchVerticesList = np.array(patchVerticesList)  # TODO: eventaully might not need this...
        manager = fold.FoldManager()
        manager.generate_h_basic_scaff(patchVerticesList[0], patchVerticesList[1], patchVerticesList[2])

        # Generate the solution from the foldManager
        solution = manager.mainFold(numHinges, numShrinks)

        # Call the keyframe funtion
        self.foldKeyframe(time, self.shapeTraverseOrder, solution, recreatePatches)

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

        numShrinks = data.inputValue(self.inNumShrinks)  # TODO: Represents maximum allowed shrinks
        numShrinks = numShrinks.asInt()

        # self.foldTest(time)
        self.foldGeneric(time, numHinges, numShrinks)

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
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inTime, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumHinges, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inNumShrinks, foldableNode.outPoint)

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
