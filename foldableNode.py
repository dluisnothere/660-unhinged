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
def getObjectVerticeNamesAndPositions(name: str) -> dict:
    # TODO: There are probably better ways by getting the vertex iterator.
    vertex_count = cmds.polyEvaluate(name, vertex=True)
    vertices = {}
    for i in range(vertex_count):
        try:
            vertex_name = "{}.vtx[{}]".format(name, i)

            # check if vertex name exists and print it:
            if cmds.objExists(vertex_name):
                print2("Vertex name exists: {}".format(vertex_name))
            vertex_translation = cmds.pointPosition(vertex_name, world=True)
            print2("Found vertex_translation!: {}".format(vertex_translation))
            vertices[vertex_name] = vertex_translation
        except Exception as e:
            print2(str(e))
            print2("Error getting vertex: {}".format(vertex_name))

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

def print2(text):
    print(text, flush=True)
    # Open a file in append mode for writing and append the text
    f = open("C:/Users/Di/Desktop/cursedPrint.txt", "a")
    f.write(text)
    f.write("\n")
    f.close()

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

    original_shapes = []

    shape_traverse_order = []
    shape_bases = []
    shape_reset_transforms = {}

    new_shapes = []
    shapes_to_delete = []

    prev_num_hinges = 0

    num_hinges = 0

    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

        # delete log file if it exists
        if os.path.exists("C:/Users/Di/Desktop/cursedPrint.txt"):
            os.remove("C:/Users/Di/Desktop/cursedPrint.txt")

        print2("Constructor called!")

    def prepareLastFrameCleanup(self):
        # For every shape in new shapes, rename it to "delete_me" + "_i"
        # and add it to the shapes_to_delete list.
        for i in range(0, len(self.new_shapes)):
            shape = self.new_shapes[i]
            new_name = "delete_me_" + str(i)
            cmds.rename(shape, new_name)
            self.shapes_to_delete.append(new_name)

        # Clear the new shapes list.
        self.new_shapes = []
    def cleanLastFrame(self):
        print("cleanLastFrame...")
        for shape in self.shapes_to_delete:
            print("cleaning shape: {}".format(shape))
            cmds.delete(shape)

        self.shapes_to_delete = []

    def setUpGenericScene(self, upper_patches, base_patch):
        # TODO: theoretically we should only need to move things in the upper patches
        # Get the transforms for each item in upper patches
        transforms = []
        for patch in upper_patches:
            transforms.append(getObjectTransformFromDag(patch))

        original_transforms = self.shape_reset_transforms
        print2("len of shape_reset_transforms {}".format(len(original_transforms)))
        print2("len of upper patches: {}".format(len(upper_patches)))
        # Set the translation for each of the patches to the original translations
        for i in range(0, len(upper_patches)):
            patch_name = upper_patches[i]
            original_translate = original_transforms[patch_name][0]
            translate_vec = OpenMaya.MVector(original_translate[0], original_translate[1], original_translate[2])
            transforms[i].setTranslation(translate_vec, OpenMaya.MSpace.kWorld)

        # Set the rotation for each of the patches to the original rotations
        for i in range(0, len(transforms)):
            patch_name = upper_patches[i]
            original_rotate = original_transforms[patch_name][1]
            print("original_rotate")
            print(original_rotate)
            print("length of original_rotate: {}".format(len(original_rotate)))
            print("original_rotate[0][0]: {}".format(original_rotate[0][0]))
            print("original_rotate[0][1]: {}".format(original_rotate[0][1]))
            print("original_rotate[0][2]: {}".format(original_rotate[0][2]))

            rotation_euler = OpenMaya.MEulerRotation(original_rotate[0][0], original_rotate[0][1], math.radians(original_rotate[0][2]))
            transforms[i].setRotation(rotation_euler)

    def generateNewPatches(self, original_patch: str, num_hinges: int):
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
            cmds.setAttr(newPatches[i] + ".rotate", originalRotation[0][0], originalRotation[0][1],
                         originalRotation[0][2])

        # Translate the patches along the direction it has been scaled in (but that is local)
        # TODO: Axis of scaling is hard coded
        originalTranslation = cmds.getAttr(original_patch + ".translate")

        # Get the world location of the bottom of the original patch
        # TODO: hard coded for the Y direction
        originalPatchBottom = originalTranslation[0][1] - originalScale * 0.5
        newPatchPositions = []
        new_transforms = []
        for i in range(0, len(newPatches)):
            newTrans = [originalTranslation[0][0], originalPatchBottom + newPatchScale * (i + 0.5),
                        originalTranslation[0][2]]
            newPatchPositions.append(newTrans)
            cmds.setAttr(newPatches[i] + ".translate", newTrans[0], newTrans[1], newTrans[2])

            # Append new patch transform to list of new transforms
            # Which will be used for its scene reset at the beginning
            new_transforms.append([newPatches[i], newTrans, originalRotation])

        # Pivot the patches.
        for i in range(0, len(newPatches)):
            # Set the pivot location to the bottom of the patch
            # TODO: check what their generated code does
            newPivot = [newPatchScale * 0.5, 0, 0]
            transform = getObjectTransformFromDag(newPatches[i])
            transform.setRotatePivot(OpenMaya.MPoint(newPivot[0], newPivot[1], newPivot[2]), OpenMaya.MSpace.kTransform,
                                     True)

        return [newPatches, new_transforms]

    # Splits the foldTest function into two parts.
    def foldKeyframe(self, time, shape_traverse_order, fold_solution, recreate_patches):
        print("Entered foldKeyframe...")

        start_angles = fold_solution.fold_transform.startAngles
        end_angles = fold_solution.fold_transform.endAngles
        num_hinges = fold_solution.modification.num_hinges

        t = time  # dictate that the end time is 90 frames hard coded for now

        # Since we are hard coded only to get 1 angel out so far, try that one
        angle = t * (end_angles[0] - start_angles[0]) / 90  # The angle we fold at this particular time is time / 90 *
        rotAxis = (0, 0, 1)

        print("angle based on t: " + str(angle))
        print("t: " + str(t))

        # Update the list of shape_traverse_order to include the new patches where the old patch was
        if (num_hinges > 0 and recreate_patches == True):
            foldable_patch = shape_traverse_order[
                0]  # TODO: make more generic, currently assumes foldable patch is at the center
            new_patches, new_transforms = self.generateNewPatches(foldable_patch, num_hinges)
            f_idx = 1

            # Remove the foldable_patch by deleting it.
            # cmds.delete(foldable_patch)
            shape_traverse_order.remove(foldable_patch)
            del self.shape_reset_transforms[foldable_patch]
            for i in range(0, len(new_patches)):
                shape_traverse_order.insert(f_idx * i, new_patches[i])
                self.shape_reset_transforms[new_patches[i]] = [new_transforms[i][1], new_transforms[i][2]]

                # Keep track of the new patches just created so we can delete it on the next iteration
                self.new_shapes.append(new_patches[i])

        # Loop through the patches and get all of their pivots.
        patchPivots = []
        for shape in shape_traverse_order:
            pivot = getObjectTransformFromDag(shape).rotatePivot(OpenMaya.MSpace.kWorld)
            patchPivots.append(pivot)
            print("Pivot: {:.6f}, {:.6f}, {:.6f}".format(pivot[0], pivot[1], pivot[2]))

        print("Setting up closest vertices...")
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
        print("Getting start and end angles...")
        print("startAngles: ")
        print(fold_solution.fold_transform.startAngles)
        print("endAngles: ")
        print(fold_solution.fold_transform.endAngles)

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

    # Fold test for non hard coded transforms: Part 1 of the logic from foldTest, calls foldKeyframe()
    def foldGeneric(self, time):

        # TODO: Should be input by the author
        self.original_shapes = ["pFoldH", "pBaseTopH"]
        self.shape_bases = ["pBaseBottomH"]

        # If self.shape_traverse_order is empty, we fill it with original shapes
        # No need to reset the scene if it hasn't been changed yet.
        if (len(self.shape_traverse_order) == 0):
            self.shape_traverse_order = self.original_shapes

            # # fill new_translation and new_rotation with original values
            for shape in self.shape_traverse_order:
                transform = getObjectTransformFromDag(shape)
                translate = transform.translation(OpenMaya.MSpace.kWorld)
                print("inserting original traslation: " + str(translate))

                rotate = transform.transformation().eulerRotation()
                print("inserting original rotation: " + str(rotate))

                self.shape_reset_transforms[shape] = [translate, [(rotate[0], rotate[1], rotate[2])]]

        else:
            # Reset the scene
            # TODO: make more generic
            self.setUpGenericScene(self.shape_traverse_order, self.shape_bases)

        foldable_patches = ["pFoldH"]

        # Hide all foldable patches from Maya
        for patch in foldable_patches:
            cmds.setAttr(patch + ".visibility", False)

        shape_vertices = []

        # First insert the pBaseBottom's vertices into shape_vertices.
        vertices_list = getObjectVerticeNamesAndPositions(self.shape_bases[0])
        shape_vertices.append(list(vertices_list.values()))

        # Repeat the procedure for the remaining patches
        for shape in self.original_shapes:
            print("Shape: {}".format(shape))
            vertices_list = getObjectVerticeNamesAndPositions(shape)
            shape_vertices.append(list(vertices_list.values()))

        shape_vertices = np.array(shape_vertices)
        print("shape_vertices:")
        print(shape_vertices)

        # Create a Fold Manager
        manager = fold.FoldManager()
        manager.generate_h_basic_scaff(shape_vertices[0], shape_vertices[1], shape_vertices[2])
        solution = manager.mainFold()

        recreate_patches = False
        if (self.num_hinges != solution.modification.num_hinges):
            self.num_hinges = solution.modification.num_hinges

            # Reset back to original shapes so you can break them again
            self.shape_traverse_order = self.original_shapes
            # self.prepareLastFrameCleanup()
            # self.cleanLastFrame()
            recreate_patches = True

        # Call the keyframe funtion
        self.foldKeyframe(time, self.shape_traverse_order, solution, recreate_patches)

    # compute
    def compute(self, plug, data):
        # Print the MDGContext
        context = data.context().isNormal()
        print2("context: " + str(context))

        if (context == False):
            print("Context is not normal, returning")
            return


        print2("compute")

        # TODO:: create the main functionality of the node. Your node should
        #         take in three floats for max position (X,Y,Z), three floats
        #         for min position (X,Y,Z), and the number of random points to
        #         be generated. Your node should output an MFnArrayAttrsData
        #         object containing the random points. Consult the homework
        #         sheet for how to deal with creating the MFnArrayAttrsData.

        timeData = data.inputValue(self.inTime)
        time = timeData.asInt()

        print2("current time: " + str(time))

        # self.foldTest(time)
        self.foldGeneric(time)

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
