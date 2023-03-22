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

# Node definition
class foldableNode(OpenMayaMPx.MPxNode):
    # Declare class variables:
    # TODO:: declare the input and output class variables
    #         i.e. inNumPoints = OpenMaya.MObject()

    # inPushDirX = OpenMaya.MObject()
    # inPushDirY = OpenMaya.MObject()
    # inPushDirZ = OpenMaya.MObject()
    #
    # inNumHinges = OpenMaya.MObject()
    # inNumShrinks = OpenMaya.MObject()
    #
    # inPushDir = OpenMaya.MObject()

    # inMeshPosX = OpenMaya.MObject()
    # inMeshPosY = OpenMaya.MObject()
    # inMeshPosZ = OpenMaya.MObject()
    # inMeshPos = OpenMaya.MObject()

    inPivX = OpenMaya.MObject()
    inPivY = OpenMaya.MObject()
    inPivZ = OpenMaya.MObject()
    inPiv = OpenMaya.MObject()

    # debug proof of concept: rotate given object around some given axis by a given degree over time
    inRotX = OpenMaya.MObject()
    inRotY = OpenMaya.MObject()
    inRotZ = OpenMaya.MObject()
    inRot = OpenMaya.MObject()

    inRotAngle = OpenMaya.MObject()

    inTime = OpenMaya.MObject() # duration of movement

    outPoint = OpenMaya.MObject() # only moving one object, so only one outpoint

    # constructor
    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

    # compute
    def compute(self, plug, data):
        # TODO:: create the main functionality of the node. Your node should
        #         take in three floats for max position (X,Y,Z), three floats
        #         for min position (X,Y,Z), and the number of random points to
        #         be generated. Your node should output an MFnArrayAttrsData
        #         object containing the random points. Consult the homework
        #         sheet for how to deal with creating the MFnArrayAttrsData.

        rotAxisData = data.inputValue(self.inRot)
        rotAxis = rotAxisData.asFloat3()

        rotAxisAngle = data.inputValue(self.inRotAngle)
        angle = rotAxisAngle.asFloat()

        pivData = data.inputValue(self.inPiv)
        piv = pivData.asFloat3()

        timeData = data.inputValue(self.inTime)
        time = timeData.asInt()

        # debug hard code translation for visibility
        pos = OpenMaya.MVector(4, 0, 0)

        pivotPoint = OpenMaya.MPoint(piv[0], piv[1], piv[2])

        #  Create a new cube if one doesn't exist
        if not cmds.objExists('rotCube1'):
            transform_node, shape_node = cmds.polyCube(n="rotCube1")

        # select the object from hierarchy
        selection_list = OpenMaya.MSelectionList()
        selection_list.add("rotCube1")

        # find the world transform from DAG. If we don't select it from the DAG,
        # we can't apply any world transforms to it
        transform_dag_path = OpenMaya.MDagPath()
        status = selection_list.getDagPath(0, transform_dag_path)
        helper = OpenMaya.MFnTransform(transform_dag_path)

        # Reset entire cube to start from clean state
        # helper.setRotation(OpenMaya.MQuaternion.identity, OpenMaya.MSpace.kObject)
        helper.setRotation(OpenMaya.MQuaternion.identity, OpenMaya.MSpace.kWorld)
        helper.setRotatePivot(OpenMaya.MPoint(0, 0, 0), OpenMaya.MSpace.kWorld, False)
        helper.setScalePivot(OpenMaya.MPoint(0, 0, 0), OpenMaya.MSpace.kWorld, False)

        # Set translate (if necessary)
        helper.setTranslation(pos, OpenMaya.MSpace.kWorld)
        # Set pivot point to the hinge
        helper.setRotatePivot(pivotPoint, OpenMaya.MSpace.kWorld, False)

        # Rotate the helper object around the pivot point
        q = OpenMaya.MQuaternion(math.radians(angle * time), OpenMaya.MVector(rotAxis[0], rotAxis[1], rotAxis[2]))
        helper.rotateBy(q, OpenMaya.MSpace.kTransform)

        data.setClean(plug)

# initializer
def nodeInitializer():
    tAttr = OpenMaya.MFnTypedAttribute()
    nAttr = OpenMaya.MFnNumericAttribute()

    # TODO:: initialize the input and output attributes. Be sure to use the
    #         MAKE_INPUT and MAKE_OUTPUT functions

    try:
        print("Initialization!\n")
        foldableNode.inPivX = nAttr.create("pivX", "px", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inPivY = nAttr.create("pivY", "py", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inPivZ = nAttr.create("pivZ", "pz", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inPiv = nAttr.create("pivPos", "pv", foldableNode.inPivX, foldableNode.inPivY,
                                            foldableNode.inPivZ)
        MAKE_INPUT(nAttr)

        foldableNode.inRotX = nAttr.create("rotX", "rx", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inRotY = nAttr.create("rotY", "ry", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inRotZ = nAttr.create("rotZ", "rz", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)

        foldableNode.inRot = nAttr.create("rotVec", "rv", foldableNode.inRotX, foldableNode.inRotY,
                                            foldableNode.inRotZ)
        MAKE_INPUT(nAttr)

        foldableNode.inRotAngle = nAttr.create("angle", "a", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)

        foldableNode.inTime = nAttr.create("inTime", "t", OpenMaya.MFnNumericData.kInt, 1)
        MAKE_INPUT(nAttr)

        foldableNode.outPoint = tAttr.create("outPoint", "oP", OpenMaya.MFnArrayAttrsData.kDynArrayAttrs)
        MAKE_OUTPUT(tAttr)

    except Exception as e:
        print(e)
        sys.stderr.write( ("Failed to create attributes of %s node\n", kPluginNodeTypeName) )

    try:
        # TODO:: add the attributes to the node and set up the
        #         attributeAffects (addAttribute, and attributeAffects)

        foldableNode.addAttribute(foldableNode.inPiv)
        foldableNode.addAttribute(foldableNode.inRot)
        foldableNode.addAttribute(foldableNode.inRotAngle)
        foldableNode.addAttribute(foldableNode.inTime)
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inPiv, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inRot, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inRotAngle, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inTime, foldableNode.outPoint)

    except Exception as e:
        print(e)
        sys.stderr.write( ("Failed to add attributes of %s node\n", kPluginNodeTypeName) )

# creator
def nodeCreator():
    return OpenMayaMPx.asMPxPtr( foldableNode() )

# initialize the script plug-in
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.registerNode( kPluginNodeTypeName, foldableNodeId, nodeCreator, nodeInitializer )
    except:
        sys.stderr.write( "Failed to register node: %s\n" % kPluginNodeTypeName )

    # Load menu
    OpenMaya.MGlobal.executeCommand("source \"" + mplugin.loadPath() + "/unhingedDialogue.mel\"")

# uninitialize the script plug-in
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)

    OpenMaya.MGlobal.executeCommand("source \"" + mplugin.loadPath() + "/removeMenu.mel\"")

    try:
        mplugin.deregisterNode( foldableNodeId )
    except:
        sys.stderr.write( "Failed to unregister node: %s\n" % kPluginNodeTypeName )