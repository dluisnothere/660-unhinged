import math

import maya.cmds as cmds
import numpy as np
import maya.OpenMaya as OpenMaya

import foldMain as fold
from typing import Dict, List, Set


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

def checkScaffoldConnectionNoErr(pivot: OpenMaya.MVector, middlepoint: OpenMaya.MVector) -> bool:
    dist = OpenMaya.MVector(pivot - middlepoint).length()
    print("Pivot distance to middle point: {:.6f}".format(dist))
    if dist > 0.0001:
        return False
    return True

def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    # print("Getting transform for {}".format(name))
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
            status = checkScaffoldConnectionNoErr(pivot, middlePoint)
            if status:
                edges.append([base, foldpatch])

    print("Edges:")
    print(edges)



class InputScaffold():
    def __init__(self, patches: List[str]):
        self.patches = patches
        self.edges = []

    def genEdges(self):
        print("implement me!")


def generateNewPatches(originalPatch: str, numHinges: int) -> (List[str], List[List[List[float]]]):
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


def foldKeyframe(time, shape_traverse_order: List[str], fold_solution):
    print("Entered foldKeyframe...")

    start_angles = fold_solution.fold_transform.startAngles
    end_angles = fold_solution.fold_transform.endAngles
    num_hinges = fold_solution.modification.num_hinges

    t = time  # dictate that the end time is 90 frames hard coded for now

    # Shrinking variables
    startPiece = fold_solution.modification.range_start
    endPiece = fold_solution.modification.range_end  # non inclusive
    numPieces = fold_solution.modification.num_pieces

    # Since we are hard coded only to get 1 angel out so far, try that one
    angle = t * (end_angles[0] - start_angles[0]) / 90  # The angle we fold at this particular time is time / 90 *
    rotAxis = (0, 0, 1)

    print("angle based on t: " + str(angle))
    print("t: " + str(t))

    # Patch shrinking
    print("shape_traverse_order: " + str(shape_traverse_order))
    # Patch splitting based on hinges
    # Update the list of shape_traverse_order to include the new patches where the old patch was
    if (num_hinges > 0):
        foldable_patch = shape_traverse_order[
            0]  # TODO: make more generic, currently assumes foldable patch is at the center
        cmds.setAttr(foldable_patch + ".visibility", False)

        for j in range(0, len(shape_traverse_order) - 1):  # every patch except last guy is foldable
            foldablePatch = shape_traverse_order[
                j]  # TODO: make more generic, currently assumes foldable patch is at the center
            shape_traverse_order.remove(foldablePatch)

            newPatches, newTransforms = generateNewPatches(foldablePatch, num_hinges)

            # Add the new patch transforms to the shape_reset_transforms and insert new patches to shape_traverse_order
            for i in range(0, len(newPatches)):
                shape_traverse_order.insert(j, newPatches[i])


    print("shape_traverse_order: " + str(shape_traverse_order))

    # Loop through the patches and get all of their pivots.
    patchPivots = []
    print("Getting all patch pivots...")
    for shape in shape_traverse_order:
        print("shape: " + shape)
        pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
        patchPivots.append(pivot)
        print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))

    closestVertices = []
    midPoints = []
    print("Get closest Vertices")
    for i in range(0, len(shape_traverse_order) - 1):
        # For each parent patch, get their vertices.
        shape = shape_traverse_order[i]
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
        middlePoint = (verticeDist * 0.5)
        print("Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePoint[0], middlePoint[1], middlePoint[2]))

        midPoints.append(middlePoint)
        # Ensure the parent and child are actually connected
        checkScaffoldConnection(childPivot, middlePoint)

    # Main Foldabilization Output generation

    # Perform rotations at once, but do not rotate the last patch
    patchTransforms = []
    print("Rotating patches...")
    for i in range(0, len(shape_traverse_order)):
        shape = shape_traverse_order[i]
        pTransform = getObjectTransformFromDag(shape)
        patchTransforms.append(pTransform)
        if (i == len(shape_traverse_order) - 1):  # TODO: fix this bc it won't work for T scaffolds
            break
        print("Now rotating shape: " + shape)
        q = OpenMaya.MQuaternion(math.radians(angle), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
        print("angle" + str(angle))
        pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)
        angle = -angle

    # Update location of closest vertices now that you've rotated each patch.
    print("Updating closest vertices and translating patches...")
    newClosestVertices = closestVertices.copy()
    newMidPoints = midPoints.copy()
    for i in range(0, len(patchPivots) - 1):
        childPivot = patchPivots[i + 1]
        for j in range(0, len(newClosestVertices[
                                  i])):  # index and use information from updated vertex positions. There should only be 2 verts here
            vertex_name, dist, vertexPoint = newClosestVertices[i][j]
            vertexPoint = cmds.pointPosition(vertex_name, world=True)
            vertexPoint = OpenMaya.MVector(vertexPoint[0], vertexPoint[1], vertexPoint[2])
            dist = OpenMaya.MVector(childPivot - vertexPoint).length()
            newClosestVertices[i][j] = (
                vertex_name, dist,
                vertexPoint)  # change my vertices to the new one, with such distance to the child pivot.

            # Print new location and distance.
            print("Closest Vertices: {}, dist: {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(newClosestVertices[i][j][0],
                                                                                      newClosestVertices[i][j][1],
                                                                                      newClosestVertices[i][j][2][
                                                                                          0],
                                                                                      newClosestVertices[i][j][2][
                                                                                          1],
                                                                                      newClosestVertices[i][j][2][
                                                                                          2]))

        verticeDistNew = newClosestVertices[i][0][2] + newClosestVertices[i][1][2]
        middlePointNew = (verticeDistNew * 0.5)
        print(
            "Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePointNew[0], middlePointNew[1], middlePointNew[2]))

        # Get the translation from the old middle point to the new middle point.
        ogMidPoint = midPoints[i]
        translation = middlePointNew - ogMidPoint
        print("Middle point translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1],
                                                                        translation[2]))

        # Translate pTop by the translation.
        childPatchTransform = patchTransforms[i + 1]
        childPatchTransform.translateBy(translation, OpenMaya.MSpace.kWorld)

    # print("shrinking patches...")
    for i in range(0, len(shape_traverse_order) - 1):
        foldable_patch = shape_traverse_order[i]
        middle = 1 / 2  # hard coded for now
        print("dealing with: " + foldable_patch)

        # Translate patch to the new midpoint
        pieceWidth = 1.0 / numPieces  # hard coded for now
        newMiddle = (startPiece + endPiece) * pieceWidth / 2

        print("newMiddle in Z direction: {}".format(newMiddle))
        transform = getObjectTransformFromDag(foldable_patch)
        translation = OpenMaya.MVector(0, 0, newMiddle) - OpenMaya.MVector(0, 0, middle)
        print("translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0],
                                       translation[1],
                                       translation[2]))
        print("translation before: {:.6f}, {:.6f}, {:.6f}".format(transform.translation(OpenMaya.MSpace.kTransform)[0],
                                         transform.translation(OpenMaya.MSpace.kTransform)[1],
                                            transform.translation(OpenMaya.MSpace.kTransform)[2]))

        transform.translateBy(translation, OpenMaya.MSpace.kTransform)

        print ("translation after: {:.6f}, {:.6f}, {:.6f}".format(transform.translation(OpenMaya.MSpace.kTransform)[0],
                                        transform.translation(OpenMaya.MSpace.kTransform)[1],
                                        transform.translation(OpenMaya.MSpace.kTransform)[2]))

        # Shrink patch by numPieces in the hard coded z direction
        shrinkFactor = (endPiece - startPiece) / numPieces
        cmds.setAttr(foldable_patch + ".scaleZ", shrinkFactor)


# Fold test for non hard coded transforms: Part 1 of the logic from foldTest
def foldGeneric():
    pushAxis = [0, -1, 0] # TODO: make a parameter

    shape_traverse_order = ["pFoldH", "pBaseTopH"] # make a list of shapes selected by user.
    shape_bases = ["pBaseBottomH"]

    shape_vertices = []

    # First insert the pBaseBottom's vertices into shape_vertices.
    vertices_list = getObjectVerticeNamesAndPositions(shape_bases[0])
    shape_vertices.append(list(vertices_list.values()))

    # Repeat the procedure for the remaining patches
    for shape in shape_traverse_order:
        print("Shape: {}".format(shape))
        vertices_list = getObjectVerticeNamesAndPositions(shape)
        shape_vertices.append(list(vertices_list.values()))

    shape_vertices = np.array(shape_vertices)
    # print("shape_vertices:")
    # print(shape_vertices)

    # Create input scaff
    patchList = shape_bases
    patchList.extend(shape_traverse_order)

    edges = getPatchConnectivity(patchList, OpenMaya.MVector(pushAxis[0], pushAxis[1], pushAxis[2]))

    # inputScaff = fold.InputScaff(patch_list)
    # Create a Fold Manager
    # manager = fold.FoldManager()
    # manager.generate_h_basic_scaff(shape_vertices[0], shape_vertices[1], shape_vertices[2])
    # solution = manager.mainFold(1, 2)

    # Call the keyframe funtion
    # foldKeyframe(60, shape_traverse_order, solution)


foldGeneric()
