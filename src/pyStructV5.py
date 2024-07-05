import numpy as np
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import math


class Node:
    """define nodes and methods"""
    nodes = []

    def __init__(self, xy=list):
        Node.nodes.append(self)
        self.id = len(Node.nodes)
        self.xy = xy
        self.fix = False
        self.loadX = 0
        self.loadY = 0
        self.forces = [0,0]

    def Fix(self):
        self.fix = True

    def AssignDx(self, x):
        self.dx = x

    def AssignDy(self, y):
        self.dy = y

    def AssignRFx(self, x):
        self.rFx = x

    def AssignRFy(self, y):
        self.rFy = y


class Member:
    """define members"""
    members = []

    def __init__(self, n1, n2, ea):
        Member.members.append(self)
        self.id = len(Member.members)
        self.n1 = n1
        self.n2 = n2
        self.ea = ea
        self.vector = Member.vector(n1, n2)
        self.gesMatrix = Member.gesM(self)
        
    def vector(n1,n2):
        """calculate length and direction of member with node1 at origin"""
        x = n2.xy[0]-n1.xy[0]   #dx
        y = n2.xy[1]-n1.xy[1]   #dy

        length = math.sqrt((x**2)+(y**2))
        dotP = np.vdot(np.array([x,y]), np.array([1,0]))   #dot product between Vector and positive horizontal axis
        
        direction = math.acos(dotP/length)  # calc direction between 0-2pi
        if y < 0:
            direction = 2*math.pi-direction

        vector = [length, direction]
        return vector
    
    def gesM(member):
        """find the Global Element Stiffness Matrix for X member"""
        theta = member.vector[1]                # get direction and length from object
        length = member.vector[0]
        # 4x4 rotation matrix of theta
        transMatrix = [[math.cos(theta)**2, math.cos(theta)*math.sin(theta), (math.cos(theta)**2)*(-1), math.cos(theta)*math.sin(theta)*(-1)],
                    [math.cos(theta)*math.sin(theta), math.sin(theta)**2, math.cos(theta)*math.sin(theta)*(-1), (math.sin(theta)**2)*(-1)],
                    [(math.cos(theta)**2)*(-1), math.cos(theta)*math.sin(theta)*(-1), math.cos(theta)**2, math.cos(theta)*math.sin(theta)],
                    [math.cos(theta)*math.sin(theta)*(-1), (math.sin(theta)**2)*(-1), math.cos(theta)*math.sin(theta), math.sin(theta)**2]]
        transArray = np.asarray(transMatrix)
        transArray = (1/length)*transArray*member.ea           # (1/L)*T*EA
        return transArray
    
    def plot(self, sp):
        x1 = self.n1.xy[0]
        y1 = self.n1.xy[1]
        x2 = self.n2.xy[0]
        y2 = self.n2.xy[1]
        sp.plot(x1,y1,x2,y2)


def splitgesM(member):
    """split 4x4 ges Matrix from X member into (4) 2x2 matrices"""
    splitV = np.vsplit(member.gesMatrix, 2)  # split vertically
    splitV1 = np.asarray(splitV[0]) 
    splitV2 = np.asarray(splitV[1])

    splitH = np.hsplit(splitV1, 2)           # split left side horizontally
    gMat11 = np.asarray(splitH[0])           
    gMat21 = np.asarray(splitH[1])
    splitH = np.hsplit(splitV2, 2)           # split right side horizontally
    gMat12 = np.asarray(splitH[0])
    gMat22 = np.asarray(splitH[1])           # quadrants assigned
    
    return gMat11, gMat12, gMat21, gMat22


def buildPssM():
    """get ges matrices from each member, split matrix, and assign to appropriate cells"""
    size = 2 * len(Node.nodes)
    pssMat = np.zeros((size,size))
    for member in Member.members:  
        g = (splitgesM(member))
        n1 = Node.nodes.index(member.n1)
        n2 = Node.nodes.index(member.n2)
        
        pssMat[2*n1][2*n1] += g[0][0][0]
        pssMat[2*n1][2*n1+1] += g[0][0][1]
        pssMat[2*n1+1][2*n1] += g[0][1][0]
        pssMat[2*n1+1][2*n1+1] += g[0][1][1]
        
        pssMat[2*n1][2*n2] += g[1][0][0]
        pssMat[2*n1][2*n2+1] += g[1][0][1]
        pssMat[2*n1+1][2*n2] += g[1][1][0]
        pssMat[2*n1+1][2*n2+1] += g[1][1][1]

        pssMat[2*n2][2*n1] += g[2][0][0]
        pssMat[2*n2][2*n1+1] += g[2][0][1]
        pssMat[2*n2+1][2*n1] += g[2][1][0]
        pssMat[2*n2+1][2*n1+1] += g[2][1][1]

        pssMat[2*n2][2*n2] += g[3][0][0]
        pssMat[2*n2][2*n2+1] += g[3][0][1]
        pssMat[2*n2+1][2*n2] += g[3][1][0]
        pssMat[2*n2+1][2*n2+1] += g[3][1][1]
    return pssMat


def buildSsM(pssMx):
    ssMx = pssMx

    i = 0
    j = 0
    while i < (len(pssMx)):
        if Node.nodes[int(i/2)].fix == True:
            split = np.split(ssMx,[j,j+1],axis=0)
            ssMx = np.concatenate((split[0],split[2]), axis=0)

            split = np.split(ssMx,[j,j+1],axis=1)
            ssMx = np.concatenate((split[0],split[2]), axis=1)
            
        else:    
            j +=1

        i +=1
    return ssMx
        

def buildForcesV():
    """build forces vector - nodes have either a known force or displacement. Internal nodes have a sum force of 0 in any given direction"""
    #nodes = nodes
    fV = []   # [[Fx1],[Fy1],[Fx2],[Fy2]...]
    fxMx = []
              # check all DOF for fixed position
    for node in Node.nodes: #check each node's DOF
        i = 2*Node.nodes.index(node)
        if node.fix == False:
            lx = node.loadX
            fxMx.append("Fx%d" % (node.id))
            fV.append(lx)
            ly = node.loadY
            fxMx.append("Fy%d" % (node.id))
            fV.append(ly)
    
    fV = np.asarray(fV,dtype=float).reshape((len(fV),1))
    fxMx = np.asarray(fxMx).reshape((len(fxMx),1))
    
    return fV,fxMx


def dMatrix():
    """build restrictions matrix - all zeros matrix with dimensions of one column, and double as many rows as there are unfixed nodes"""
    uMatrix = []
    for node in Node.nodes:
        if node.fix == False:

            uMatrix.append("Ux" + str((Node.nodes.index(node))+1))
            uMatrix.append("Uy" + str((Node.nodes.index(node))+1))

    uMatrix = np.asarray(uMatrix).reshape((len(uMatrix),1))
    dMatrix = np.empty((len(uMatrix),1))

    i = 0
    while i < len(uMatrix)-1:
            dMatrix[i][0] = float(0)
            dMatrix[i+1][0] = float(0)

            i +=1
            

    return dMatrix, uMatrix   


def assignDisplacements(dV):
    i = 0
    for f in dV:
        node = Node.nodes[int(f[1][2])-1]
        if f[1][1] == "x":
            node.AssignDx(dV[i][0])
        elif f[1][1] == "y":
            node.AssignDy(dV[i][0])
        else:
            ValueError
        i+=1

    for node in Node.nodes:
        try:
            if node.dx == float:
                pass
        except AttributeError:
            node.AssignDx(0)
        
        try:
            if node.dy == float:
                pass
        except AttributeError:
            node.AssignDy(0)


def assignReactionForces(rfV):
    i = 0
    for f in rfV:
        node = Node.nodes[int(i/2)]
        if i%2 == 0: #x values
            node.AssignRFx(rfV[i])
        elif i%2 == 1: #y values
            node.AssignRFy(rfV[i])
        else:
            ValueError
        i+=1


def rebuildDMx():
    mx = np.zeros((2*len(Node.nodes)))
    for node in Node.nodes:
        if node.dx != 0:
            mx[2*int(node.id)-2] = node.dx
        if node.dy != 0:
            mx[2*int(node.id)-1] = node.dy
    mx = np.reshape(mx, (len(mx),1))
    return mx


def rebuildFMx(): 
    mx = []
    for node in Node.nodes:
        try:
            mx.append(node.rFx)
        except:
            mx.append(node.loadX)
        try:
            mx.append(node.rFy)
        except:
            mx.append(node.loadY)
    mx = np.asarray(mx).reshape((len(mx),1))

    return mx


def plotMechanism():
    pltmargin = 5
    minX = 0
    maxX = 0
    minY = 0
    maxY = 0

    longest = 0
    for member in Member.members:   #find length of longest member
        if member.vector[0] > longest:
            longest = member.vector[0]
            pltmargin = longest*.2

    for node in Node.nodes: # set min and max values for axes
        if node.xy[0] <=minX:
            minX = node.xy[0]-pltmargin
        if node.xy[1] <=minY:
            minY = node.xy[1]-pltmargin
        if node.xy[0] >=maxX:
            maxX = node.xy[0]+pltmargin
        if node.xy[1] >=maxY:
            maxY = node.xy[1]+pltmargin
    
    

    fig = plt.figure(figsize=(8, 4))    # setup figure and plot
    sp = fig.add_subplot(autoscale_on=False, xlim=(minX, maxX), ylim=(minY, maxY))
    sp.set_aspect('equal')
    sp.set_axis_off()
    #sp.grid()

    for member in Member.members:   # draw members
        sp.plot([member.n1.xy[0],member.n2.xy[0]],[member.n1.xy[1],member.n2.xy[1]])
        
    triSize = 10
    for node in Node.nodes: 
        x = node.xy[0]
        y = node.xy[1]
        sp.text(x, y, node.id)
        if node.fix == True:    # draw triangles under fixed nodes
            codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*2 + [mpath.Path.CLOSEPOLY]
            vertices = [(x, y), (x+triSize, y-triSize), (x-triSize, y-triSize), (x, y)]
            path = mpath.Path(vertices, codes)
            patch = patches.PathPatch(path)
            sp.add_patch(patch)

        scale = .01
        w = 15

        if node.loadX != 0:  #draw load arrows in yellow
            xAro = patches.Arrow(node.xy[0],node.xy[1], scale*node.loadX, 0, width=w, color="y")
            sp.add_patch(xAro)
        if node.loadY != 0:
            yAro = patches.Arrow(node.xy[0],node.xy[1], 0, scale*node.loadY, width=w, color="y")
            sp.add_patch(yAro)


        try:    #draw reaction forces in blue
            if node.rFx != 0:
                if node.loadX == 0:
                    
                    xAro = patches.Arrow((node.xy[0]-scale*node.rFx), node.xy[1], scale*node.rFx, 0, width=w, color="b")
                    sp.add_patch(xAro)
        except:
            pass
        try:
            if node.rFy != 0:
                if node.loadY == 0:
                    yAro = patches.Arrow(node.xy[0], (node.xy[1]-scale*node.rFy), 0, scale*node.rFy, width=w, color="b")
                    sp.add_patch(yAro)
        except:
            pass
    plt.show()


def main():
    pssMx = buildPssM() # build PrimarySstructure Stiffness Matrix 
    ssMx = buildSsM(pssMx) # (k) restrict pssMx based on fixed nodes
    uV = dMatrix() # (q) build zeros Lx1 matrix; L = 2*(#number of fixed nodes)
    fV = buildForcesV() # (f) build zeros Lx1 matrix; L = 2*(# of nodes with applied loads)

    # k*q=f
    dV = np.linalg.solve(ssMx, fV[0])

    loads = np.hstack((fV[1],fV[0]))
    displacements = np.round(dV,5)

    dV = np.hstack((dV, uV[1]))
    displacements = np.hstack((displacements, uV[1]))
    assignDisplacements(dV)
    dMx = rebuildDMx()

    rfV = np.matmul(pssMx,dMx)
    assignReactionForces(rfV)
    fMx = rebuildFMx()

    plotMechanism() #plot linkage

"""Nodes and Members"""

node1 = Node([0,0])
node2 = Node([125,0])
node3 = Node([61.36039,63.63961])
node4 = Node([-63.63961,63.63961])
node5 = Node([25,25])
node6 = Node([250,-5])
node7 = Node([-28.69749,61.62758])
node8 = Node([-40.62708,119.05101])
node9 = Node([-157.60774,75])

A = Member(node1, node2, 2*10**7)
B = Member(node2, node3, 2*10**7)
C = Member(node5, node7, 2*10**7)
D = Member(node3, node4, 2*10**7)
E = Member(node1, node5, 2*10**7)
F = Member(node5, node2, 2*10**7)
G = Member(node2, node6, 2*10**7)
H = Member(node6, node3, 2*10**7)
I = Member(node4, node7, 2*10**7)
J = Member(node7, node8, 2*10**7)
K = Member(node8, node4, 2*10**7)
L = Member(node8, node9, 2*10**7)

node1.Fix()
node4.Fix()
node9.Fix()

node6.loadY = 1000

main()
