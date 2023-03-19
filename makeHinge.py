import sys

# Imports to use the Maya Python API
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

# Import the Python wrappers for MEL commands
import maya.cmds as cmds

# The name of the command. 
kPluginCmdName = "addHinge"

nameFlag = "-n"
nameLongFlag = "-name"
idFlag = "-id"
idLongFlag = "-identification"

xtFlag = "-xt"
xtFlagLong = "-x_translation"
xrFlag = "-xr"
xrFlagLong = "-x_rotation"
ytFlag = "-yt"
ytFlagLong = "-y_translation"
yrFlag = "-yr"
yrFlagLong = "-y_rotation"
ztFlag = "-zt"
ztFlagLong = "-z_translation"
zrFlag = "-zr"
zrFlagLong = "-z_rotation"

class helloMayaCommand(OpenMayaMPx.MPxCommand):
    cylinder_counter = 0
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def addHinge(self, x_axis, y_axis, z_axis, x_pos, y_pos, z_pos, name):
        cmds.cylinder(axis=[x_axis, y_axis, z_axis], p=[x_pos, y_pos, z_pos], n=name)

    def doIt(self, argList):
        # TODO fill in this code to implement the command.
        xt = 0.0
        xr = 1.0
        yt = 0.0
        yr = 0.0
        zt = 0.0
        zr = 0.0
        argData = OpenMaya.MArgDatabase(self.syntax(), argList)
        if argData.isFlagSet(xtFlag):
            xt = argData.flagArgumentFloat(xtFlag, 0)
        if argData.isFlagSet(xrFlag):
            xr = argData.flagArgumentFloat(xrFlag, 0)
        
        name = "randomName" + str(self.cylinder_counter)
        self.cylinder_counter += 1
        self.addHinge(xr, yr, zr, xt, yt, zt, name)
        self.setResult("Executed command")

        # name_str = "anon"
        # id_str = "0"
        # argData = OpenMaya.MArgDatabase(self.syntax(), argList)
        # if argData.isFlagSet(nameFlag):
        #     name_str = argData.flagArgumentString(nameFlag, 0)
        # if argData.isFlagSet(idFlag):
        #     id_str = argData.flagArgumentString(idFlag, 0)
        
        # output = "Name: " + name_str + " ID: " + id_str
        # cmds.confirmDialog( title='Hello Maya!', message=output, button=['Done'], defaultButton='Done', cancelButton='Done', dismissString='Done' )
        # self.setResult("Executed command")

# Create an instance of the command.
def cmdCreator():
    return OpenMayaMPx.asMPxPtr(helloMayaCommand())

# Syntax Creator
def syntaxCreator():
    syntax = OpenMaya.MSyntax()
    # syntax.addFlag(nameFlag, nameLongFlag, OpenMaya.MSyntax.kString)
    # syntax.addFlag(idFlag, idLongFlag, OpenMaya.MSyntax.kString)
    syntax.addFlag(xtFlag, xtFlagLong, OpenMaya.MSyntax.kDouble)
    syntax.addFlag(xrFlag, xrFlagLong, OpenMaya.MSyntax.kDouble)
    return syntax 

# Initialize the plugin
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject, "cg@penn", "1.0", "2012")
    try:
        mplugin.registerCommand(kPluginCmdName, cmdCreator, syntaxCreator)
    except:
        sys.stderr.write("Failed to register command: %s\n" % kPluginCmdName)
        raise

# Uninitialize the plugin
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(kPluginCmdName)
    except:
        sys.stderr.write("Failed to unregister command: %s\n" % kPluginCmdName)
        raise
