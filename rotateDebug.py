import maya.OpenMaya as OpenMaya
import math
import maya.cmds as cmds

def getObjectTransformFromDag(name: str) -> OpenMaya.MFnTransform:
    selection_list = OpenMaya.MSelectionList()
    selection_list.add(name)
    transform_dag_path = OpenMaya.MDagPath()
    status = selection_list.getDagPath(0, transform_dag_path)
    return OpenMaya.MFnTransform(transform_dag_path)


trans = getObjectTransformFromDag("pFoldH")
rotation_euler = OpenMaya.MEulerRotation()
trans.getRotation(rotation_euler)

print("Rotation: {:.6f},{:.6f},{:.6f}".format(math.degrees(rotation_euler.x), math.degrees(rotation_euler.y), math.degrees(rotation_euler.z)))
exit(0)

# DEBUG: print out the current transform before rotation
vertex0 = cmds.pointPosition("pFoldH.vtx[0]", world=True)
vertex1 = cmds.pointPosition("pFoldH.vtx[1]", world=True)
vertex2 = cmds.pointPosition("pFoldH.vtx[2]", world=True)
vertex3 = cmds.pointPosition("pFoldH.vtx[3]", world=True)

print("p-points0: {:.6f},{:.6f},{:.6f}".format(vertex0[0], vertex0[1], vertex0[2]))
print("p-points2: {:.6f},{:.6f},{:.6f}".format(vertex2[0], vertex2[1], vertex2[2]))

pTransform = getObjectTransformFromDag("pFoldH")
q = OpenMaya.MQuaternion(math.radians(1.0), OpenMaya.MVector(0,0,1))
pTransform.rotateBy(q, OpenMaya.MSpace.kTransform)

# Query again
pTransformAfter = getObjectTransformFromDag("pFoldH")
vertex_translation0 = cmds.pointPosition("pFoldH.vtx[0]", world=True)
vertex_translation1 = cmds.pointPosition("pFoldH.vtx[1]", world=True)
vertex_translation2 = cmds.pointPosition("pFoldH.vtx[2]", world=True)
vertex_translation3 = cmds.pointPosition("pFoldH.vtx[3]", world=True)

print("p-points0: {:.6f},{:.6f},{:.6f}".format(vertex_translation0[0],
                                                     vertex_translation0[1],
                                                     vertex_translation0[2]))

print("p-points2: {:.6f},{:.6f},{:.6f}".format(vertex_translation2[0],
                                                     vertex_translation2[1],
                                                     vertex_translation2[2]))