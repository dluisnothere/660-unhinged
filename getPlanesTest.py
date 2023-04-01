import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import numpy as np


def isPolyPlane(obj):
    # Create an MSelectionList object
    selList = OpenMaya.MSelectionList()
    selList.add(obj)

    # Get the MDagPath of the object
    transformDagPath = OpenMaya.MDagPath()
    status = selList.getDagPath(0, transformDagPath)

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



# Test to see if selected objects are all planes
list = cmds.ls(selection=True)

for obj in list:
    print("Checking " + obj)
    if isPolyPlane(obj):
        print(obj + " is a plane")
    else:
        print(obj + " is NOT a plane")
