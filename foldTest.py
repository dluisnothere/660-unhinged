import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import math


# Resets the whole scene to the original layout, undoing any translation or rotation.
# Does not affect pivot points.
def setUpScene():
    pMiddle = getObjectTransformFroMDag("pMiddle")
    pBottom = getObjectTransformFroMDag("pBottom")
    pTop = getObjectTransformFroMDag("pTop")

    # Set the rotation for both to 0, 0, -90. Keep in mind that EulerRotation is in radians.
    pMiddle.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pBottom.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))
    pTop.setRotation(OpenMaya.MEulerRotation(0, 0, math.radians(-90)))

    # Set the translation for pBottom to 0, 0, 0 and pTop to 0, 1, 0.
    pBottom.setTranslation(OpenMaya.MVector(0, 0, 0), OpenMaya.MSpace.kWorld)
    pTop.setTranslation(OpenMaya.MVector(0, 2, 0), OpenMaya.MSpace.kWorld)
    pMiddle.setTranslation(OpenMaya.MVector(0, 1, 0), OpenMaya.MSpace.kWorld)


def getObjectTransformFroMDag(name: str) -> OpenMaya.MFnTransform:
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


# Reset everything.
setUpScene()

t = 0
shape_traverse_order = ["pBottom", "pMiddle", "pTop"]

# Loop through the patches and get all of their pivots.
patchPivots = []
for shape in shape_traverse_order:
    pivot = getObjectTransformFroMDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
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
    print("current closest len: ", len(currentClosest))
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

t = 2
angle = 5
rotAxis = (0, 0, 1)

# Perform rotations at once
patchTransforms = []
for i in range(0, len(shape_traverse_order)):
    shape = shape_traverse_order[i]
    pTransform = getObjectTransformFroMDag(shape)
    patchTransforms.append(pTransform)
    q = OpenMaya.MQuaternion(math.radians(angle * t), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
    pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)
    angle = -angle

# Update location of closest vertices now that you've rotated each patch.
newClosestVertices = closestVertices.copy()
newMidPoints = midPoints.copy()
for i in range(0, len(patchPivots) - 1):
    childPivot = patchPivots[i + 1]
    for j in range(0, len(newClosestVertices[i])): # index and use information from updated vertex positions. There should only be 2 verts here
        vertex_name, dist, vertexPoint = newClosestVertices[i][j]
        vertexPoint = cmds.pointPosition(vertex_name, world=True)
        vertexPoint = OpenMaya.MVector(vertexPoint[0], vertexPoint[1], vertexPoint[2])
        dist = OpenMaya.MVector(childPivot - vertexPoint).length()
        newClosestVertices[i][j] = (vertex_name, dist, vertexPoint) # change my vertices to the new one, with such distance to the child pivot.

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


