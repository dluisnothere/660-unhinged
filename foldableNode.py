import sys
import random

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

    # debug proof of concept: rotate given object around some given axis by a given degree over time
    # inRotX = OpenMaya.MObject()
    # inRotY = OpenMaya.MObject()
    # inRotZ = OpenMaya.MObject()
    # inRot = OpenMaya.MObject()
    #
    # inRotAngle = OpenMaya.MObject()

    ## debug proof of concept: move object around
    inTransX = OpenMaya.MObject()
    inTransY = OpenMaya.MObject()
    inTransZ = OpenMaya.MObject()
    inTrans = OpenMaya.MObject()

    inStepSize = OpenMaya.MObject()

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
        if (plug == self.outPoint):
            # # get all handles for input attributes
            # pushDirData = data.inputValue(self.inPushDir)
            # pushDir = pushDirData.asFloat3()
            #
            # numHingesData = data.inputValue(self.inNumHinges)
            # numHinges = numHingesData.asLong()
            #
            # numShrinkData = data.inputValue(self.inNumShrinks)
            # numShrink = numShrinkData.asLong()

            # meshPosData = data.inputValue(self.inMeshPos)
            # pos = meshPosData.asFloat3()
            #
            # rotAxisData = data.inputValue(self.inRot)
            # rotAxis = rotAxisData.asFloat3()
            #
            # rotAxisAngle = data.inputValue(self.inRotAngle)
            # angle = rotAxisAngle.asFloat()

            transData = data.inputValue(self.inTrans)
            translate = transData.asFloat3()

            timeData = data.inputValue(self.inTime)
            time = timeData.asInt()

            stepSizeData = data.inputValue(self.inStepSize)
            stepSize = stepSizeData.asFloat()

            pointData = data.outputValue(foldableNode.outPoint)  # the MDataHandle
            pointAAD = OpenMaya.MFnArrayAttrsData()  # the MFnArrayAttrsData
            pointObject = pointAAD.create()  # the MObject

            # Create the vectors for “position” and “id”. Names and types must match
            # the table above.
            positionArray = pointAAD.vectorArray("position")
            # translationArray = pointAAD.vectorArray("translation")
            # rotationArray = pointAAD.vectorArray("rotation")
            # each point has ID
            # idArray = pointAAD.doubleArray("id")

            # generate position outputs over time, similar to processPy\
            pos = OpenMaya.MVector(0, 0, 0)
            # rot = OpenMaya.MVector(0, 0, 0)

            for i in range(0, time):
                pos += OpenMaya.MVector(stepSize * translate[0], stepSize * translate[1], stepSize * translate[2])
                # rot += OpenMaya.MVector(angle * rotAxis[0], angle * rotAxis[1], angle * rotAxis[2])

            print("pos: ")
            print(pos)
            positionArray.append(pos)

            # rotationArray.append(rot)
            #
            # translation = pos - rot
            # trans = OpenMaya.MVector(translation[0], translation[1], translation[2])
            # translationArray.append(trans)

            # Finally set the output data handle pointsData.setMObject(pointsObject)
            pointData.setMObject(pointObject)

        data.setClean(plug)

# initializer
def nodeInitializer():
    tAttr = OpenMaya.MFnTypedAttribute()
    nAttr = OpenMaya.MFnNumericAttribute()

    # TODO:: initialize the input and output attributes. Be sure to use the
    #         MAKE_INPUT and MAKE_OUTPUT functions

    try:
        print("Initialization!\n")
        foldableNode.inTransX = nAttr.create("transX", "tx", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inTransY = nAttr.create("transY", "ty", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        foldableNode.inTransZ = nAttr.create("transZ", "tz", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)

        foldableNode.inStepSize = nAttr.create("step", "s", OpenMaya.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)

        foldableNode.inTrans = nAttr.create("transVec", "it", foldableNode.inTransX, foldableNode.inTransY,
                                            foldableNode.inTransZ)
        MAKE_INPUT(nAttr)

        # foldableNode.inMeshPosX = nAttr.create("posX", "px", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        # foldableNode.inMeshPosY = nAttr.create("posY", "py", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        # foldableNode.inMeshPosZ = nAttr.create("posZ", "pz", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        #
        # foldableNode.inMeshPos = nAttr.create("posVec", "pv", foldableNode.inMeshPosX, foldableNode.inMeshPosY,
        #                                       foldableNode.inMeshPosZ)
        # MAKE_INPUT(nAttr)
        #
        # foldableNode.inRotX = nAttr.create("rotX", "rx", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        # foldableNode.inRotY = nAttr.create("rotY", "ry", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        # foldableNode.inRotZ = nAttr.create("rotZ", "rz", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)
        #
        # foldableNode.inRotAxis = nAttr.create("rotVec", "rv", foldableNode.inRotX, foldableNode.inRotY,
        #                                     foldableNode.inRotZ)
        # MAKE_INPUT(nAttr)
        #
        # foldableNode.inRotAngle = nAttr.create("angle", "a", OpenMaya.MFnNumericData.kFloat)
        # MAKE_INPUT(nAttr)

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

        foldableNode.addAttribute(foldableNode.inTrans)
        foldableNode.addAttribute(foldableNode.inStepSize)

        # foldableNode.addAttribute(foldableNode.inRotAxis)
        # foldableNode.addAttribute(foldableNode.inRotAngle)
        foldableNode.addAttribute(foldableNode.inTime)
        foldableNode.addAttribute(foldableNode.outPoint)

        foldableNode.attributeAffects(foldableNode.inTrans, foldableNode.outPoint)
        foldableNode.attributeAffects(foldableNode.inStepSize, foldableNode.outPoint)
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
    try:
        mplugin.deregisterNode( foldableNodeId )
    except:
        sys.stderr.write( "Failed to unregister node: %s\n" % kPluginNodeTypeName )