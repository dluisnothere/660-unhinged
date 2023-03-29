import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMaya as OpenMaya
import math
import foldMain as fold
import numpy as np


# Resets the whole scene to the original layout, undoing any translation or rotation.
def setUpVertScene():
    pMiddle = getObjectTransformFromDag("pMiddle")
    pMiddle2 = getObjectTransformFromDag("pMiddle2")
    pBottom = getObjectTransformFromDag("pBottom")
    pTop = getObjectTransformFromDag("pTop")

    pBaseTop = getObjectTransformFromDag("pBaseTop")
    pBaseBottom = getObjectTransformFromDag("pBaseBottom")

    # Set the translation for pBottom to 0, 0, 0 and pTop to 0, 1, 0.
    pBottom.setTranslation(OpenMaya.MVector(3, 0, 0), OpenMaya.MSpace.kWorld)
    pTop.setTranslation(OpenMaya.MVector(3, 3, 0), OpenMaya.MSpace.kWorld)
    pMiddle.setTranslation(OpenMaya.MVector(3, 1, 0), OpenMaya.MSpace.kWorld)
    pMiddle2.setTranslation(OpenMaya.MVector(3, 2, 0), OpenMaya.MSpace.kWorld)

    pBaseTop.setTranslation(OpenMaya.MVector(3, 3.5, 0), OpenMaya.MSpace.kWorld)
    pBaseBottom.setTranslation(OpenMaya.MVector(3, -0.5, 0), OpenMaya.MSpace.kWorld)

    # # Set the rotation for both to 0, 0, -90. Keep in mind that EulerRotation is in radians.
    pMiddle.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pMiddle2.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pBottom.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pTop.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))

    pBaseTop.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(0)))
    pBaseBottom.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(0)))


def setUpHorizScene():
    pLeft = getObjectTransformFromDag("pLeft")
    pRight = getObjectTransformFromDag("pRight")
    pHMid = getObjectTransformFromDag("pHMid")
    pHMid2 = getObjectTransformFromDag("pHMid2")

    # Set the translation for pBottom to 0, 0, 0 and pTop to 0, 1, 0.
    pLeft.setTranslation(OpenMaya.MVector(-3, 0, 0), OpenMaya.MSpace.kWorld)
    pRight.setTranslation(OpenMaya.MVector(-3, 0, 3), OpenMaya.MSpace.kWorld)
    pHMid.setTranslation(OpenMaya.MVector(-3, 0, 1), OpenMaya.MSpace.kWorld)
    pHMid2.setTranslation(OpenMaya.MVector(-3, 0, 2), OpenMaya.MSpace.kWorld)

    # # Set the rotation for both to 0, 0, -90. Keep in mind that EulerRotation is in radians.
    pLeft.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pRight.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pHMid.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pHMid2.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))


def setUpVertBasicScene():
    print("setUpVertScene: BASIC")
    pFold = getObjectTransformFromDag("pFoldH")
    pBaseTop = getObjectTransformFromDag("pBaseTopH")
    pBaseBottom = getObjectTransformFromDag("pBaseBottomH")

    # Set the translation for pBottom to 0, 0, 0 and pTop to 0, 1, 0.
    pFold.setTranslation(OpenMaya.MVector(0, 1.5, 5), OpenMaya.MSpace.kWorld)
    pBaseTop.setTranslation(OpenMaya.MVector(0, 2, 5), OpenMaya.MSpace.kWorld)
    pBaseBottom.setTranslation(OpenMaya.MVector(0, 1, 5), OpenMaya.MSpace.kWorld)

    # # Set the rotation for both to 0, 0, -90. Keep in mind that EulerRotation is in radians.
    pFold.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pBaseTop.setRotation(OpenMaya.MEulerRotation(0, 0, 0))
    pBaseBottom.setRotation(OpenMaya.MEulerRotation(0, 0, 0))


def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)
    return OpenMaya.MFnTransform(transform_dag_path)


# Returns a dictionary of names and positions in world space.
def getObjectVerticeNamesAndPositions(name: str) -> dict:
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


def foldVertScene():
    t = 0
    shape_traverse_order = ["pBottom", "pMiddle", "pMiddle2", "pTop", "pBaseTop"]
    shape_bases = ["pBaseBottom", "pBaseTop"]

    # Loop through the patches and get all of their pivots.
    patchPivots = []
    for shape in shape_traverse_order:
        pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
        patchPivots.append(pivot)
        print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))

    closestVertices = []
    midPoints = []
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

    t = 70
    angle = 1
    rotAxis = (0, 0, 1)

    # Perform rotations at once, but do not rotate the last patch
    patchTransforms = []
    for i in range(0, len(shape_traverse_order)):
        shape = shape_traverse_order[i]
        pTransform = getObjectTransformFromDag(shape)
        patchTransforms.append(pTransform)
        if (i == len(shape_traverse_order) - 1):
            break
        q = OpenMaya.MQuaternion(math.radians(angle * t), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
        pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)
        angle = -angle

    # Update location of closest vertices now that you've rotated each patch.
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
                                                                                      newClosestVertices[i][j][2][0],
                                                                                      newClosestVertices[i][j][2][1],
                                                                                      newClosestVertices[i][j][2][2]))

        verticeDistNew = newClosestVertices[i][0][2] + newClosestVertices[i][1][2]
        middlePointNew = (verticeDistNew * 0.5)
        print("Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePointNew[0], middlePointNew[1], middlePointNew[2]))

        # Get the translation from the old middle point to the new middle point.
        ogMidPoint = midPoints[i]
        translation = middlePointNew - ogMidPoint
        print("Middle point translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1], translation[2]))

        # Translate pTop by the translation.
        childPatchTransform = patchTransforms[i + 1]
        childPatchTransform.translateBy(translation, OpenMaya.MSpace.kWorld)


def foldHorizScene():
    t = 0
    shape_traverse_order = ["pRight", "pHMid2", "pHMid", "pLeft"]

    # Loop through the patches and get all of their pivots.
    patchPivots = []
    for shape in shape_traverse_order:
        pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
        patchPivots.append(pivot)
        print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))

    closestVertices = []
    midPoints = []
    for i in range(0, len(shape_traverse_order) - 1):
        # For each parent patch, get their vertices.
        shape = shape_traverse_order[i]
        bottomVertices = getObjectVerticeNamesAndPositions(shape)

        childPivot = patchPivots[i + 1]
        print("Child's Pivot: {:.6f}, {:.6f}, {:.6f}".format(childPivot[0], childPivot[1], childPivot[2]))

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

    t = 10
    angle = 5
    rotAxis = (0, 1, 0)

    # Perform rotations at once
    patchTransforms = []
    for i in range(0, len(shape_traverse_order)):
        shape = shape_traverse_order[i]
        pTransform = getObjectTransformFromDag(shape)
        patchTransforms.append(pTransform)
        q = OpenMaya.MQuaternion(math.radians(angle * t), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
        pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)
        angle = -angle

    # Update location of closest vertices now that you've rotated each patch.
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
                                                                                      newClosestVertices[i][j][2][0],
                                                                                      newClosestVertices[i][j][2][1],
                                                                                      newClosestVertices[i][j][2][2]))

        verticeDistNew = newClosestVertices[i][0][2] + newClosestVertices[i][1][2]
        middlePointNew = (verticeDistNew * 0.5)
        print("Middle Point: {:.6f}, {:.6f}, {:.6f}".format(middlePointNew[0], middlePointNew[1], middlePointNew[2]))

        # Get the translation from the old middle point to the new middle point.
        ogMidPoint = midPoints[i]
        translation = middlePointNew - ogMidPoint
        print("Middle point translation: {:.6f}, {:.6f}, {:.6f}".format(translation[0], translation[1], translation[2]))

        # Translate pTop by the translation.
        childPatchTransform = patchTransforms[i + 1]
        childPatchTransform.translateBy(translation, OpenMaya.MSpace.kWorld)


def generateNewPatches(original_patch: str, num_hinges: int):
    # Compute the new patch scale values based on original_patch's scale and num_patches
    # TODO: Hard coded for split in the x Direction, but need to be more general later on.
    numPatches = num_hinges + 1
    originalScale = cmds.getAttr(original_patch + ".scaleX")
    newPatchScale = originalScale / numPatches

    # Generate new patches.
    newPatches = []
    for i in range(0, numPatches):
        # TODO: Based on the axis we shrink, either width or height will be the original patch's scale
        # This command generates a new polyplane in the scene
        newPatch = cmds.polyPlane(name=original_patch + "_" + str(i), width=newPatchScale, height=originalScale,
                                  subdivisionsX=1,
                                  subdivisionsY=1)
        newPatches.append(newPatch[0])

    # Rotate the new patches with the same rotation as the original_patch
    originalRotation = cmds.getAttr(original_patch + ".rotate")
    for i in range(0, len(newPatches)):
        cmds.setAttr(newPatches[i] + ".rotate", originalRotation[0][0], originalRotation[0][1], originalRotation[0][2])

    # Translate the patches along the direction it has been scaled in (but that is local)
    # TODO: Axis of scaling is hard coded
    originalTranslation = cmds.getAttr(original_patch + ".translate")

    # Get the world location of the bottom of the original patch
    # TODO: hard coded for the Y direction
    originalPatchBottom = originalTranslation[0][1] - originalScale * 0.5
    newPatchPositions = []
    for i in range(0, len(newPatches)):
        newTrans = [originalTranslation[0][0], originalPatchBottom + newPatchScale * (i + 0.5),
                    originalTranslation[0][2]]
        newPatchPositions.append(newTrans)
        cmds.setAttr(newPatches[i] + ".translate", newTrans[0], newTrans[1], newTrans[2])

    # Pivot the patches.
    for i in range(0, len(newPatches)):
        # Set the pivot location to the bottom of the patch
        # TODO: check what their generated code does
        newPivot = [newPatchScale * 0.5, 0, 0]
        transform = getObjectTransformFromDag(newPatches[i])
        transform.setRotatePivot(OpenMaya.MPoint(newPivot[0], newPivot[1], newPivot[2]), OpenMaya.MSpace.kTransform, True)

    return newPatches


def foldKeyframe(time, shape_traverse_order, fold_solution):
    print("Entered foldKeyframe...")

    start_angles = fold_solution.fold_transform.startAngles
    end_angles = fold_solution.fold_transform.endAngles
    num_hinges = fold_solution.modification.num_hinges

    # Patch shrinking parameters
    num_pieces = fold_solution.modification.num_pieces
    start_piece = fold_solution.modification.range_start
    end_piece = fold_solution.modification.range_end

    t = time  # dictate that the end time is 90 frames hard coded for now

    # Since we are hard coded only to get 1 angel out so far, try that one
    angle = t * (end_angles[0] - start_angles[0]) / 90  # The angle we fold at this particular time is time / 90 *
    rotAxis = (0, 0, 1)

    print("angle based on t: " + str(angle))
    print("t: " + str(t))

    # Patch shrinking


    # Patch splitting based on hinges
    # Update the list of shape_traverse_order to include the new patches where the old patch was
    if (num_hinges > 0):
        foldable_patch = shape_traverse_order[
            0]  # TODO: make more generic, currently assumes foldable patch is at the center
        new_patches = generateNewPatches(foldable_patch, num_hinges)
        f_idx = 1

        # Remove the foldable_patch by deleting it.
        cmds.delete(foldable_patch)
        shape_traverse_order.remove(foldable_patch)

        for i in range(0, len(new_patches)):
            shape_traverse_order.insert(f_idx * i, new_patches[i])

    # Loop through the patches and get all of their pivots.
    patchPivots = []
    for shape in shape_traverse_order:
        pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
        patchPivots.append(pivot)
        print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))

    closestVertices = []
    midPoints = []
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


# Fold test for non hard coded transforms: Part 1 of the logic from foldTest
def foldGeneric():
    shape_traverse_order = ["pFoldH", "pBaseTopH"]
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
    print("shape_vertices:")
    print(shape_vertices)

    # Create a Fold Manager
    manager = fold.FoldManager()
    manager.generate_h_basic_scaff(shape_vertices[0], shape_vertices[1], shape_vertices[2])
    solution = manager.mainFold(1)

    # Call the keyframe funtion
    foldKeyframe(60, shape_traverse_order, solution)


# setUpVertBasicScene()
# foldGeneric()