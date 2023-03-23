import sys
import random
import math

import maya.OpenMaya as OpenMaya
import maya.OpenMayaAnim as OpenMayaAnim
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds


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


# Static functions
def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)
    return OpenMaya.MFnTransform(transform_dag_path)


def setUpVertScene():
    print("setUpVertScene")
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


# Node definition
class foldableNode(OpenMayaMPx.MPxNode):
    # Declare class variables:
    # TODO:: declare the input and output class variables
    #         i.e. inNumPoints = OpenMaya.MObject()

    # duration of movement
    inTime = OpenMaya.MObject()

    # Dummy output plug that can be connected to the input of an instancer node
    # so our node can "live" somewhere.
    outPoint = OpenMaya.MObject()

    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

    # fold
    def fold(self, time):
        setUpVertScene()

        shape_traverse_order = ["pBottom", "pMiddle", "pMiddle2", "pTop", "pBaseTop"]

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

        t = time
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

    # compute
    def compute(self, plug, data):
        print("compute")
        # TODO:: create the main functionality of the node. Your node should
        #         take in three floats for max position (X,Y,Z), three floats
        #         for min position (X,Y,Z), and the number of random points to
        #         be generated. Your node should output an MFnArrayAttrsData
        #         object containing the random points. Consult the homework
        #         sheet for how to deal with creating the MFnArrayAttrsData.

        timeData = data.inputValue(self.inTime)
        time = timeData.asInt()

        self.fold(time)

        data.setClean(plug)


# initializer
def nodeInitializer():
    tAttr = OpenMaya.MFnTypedAttribute()
    nAttr = OpenMaya.MFnNumericAttribute()

    # TODO:: initialize the input and output attributes. Be sure to use the
    #         MAKE_INPUT and MAKE_OUTPUT functions

    try:
        print("Initialization!\n")
        foldableNode.inTime = nAttr.create("inTime", "t", OpenMaya.MFnNumericData.kInt, 1)
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
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inTime, foldableNode.outPoint)

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
