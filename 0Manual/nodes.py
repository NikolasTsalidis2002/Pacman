import pygame
from vector import Vector2
from constants import *
import numpy as np

class Node(object):
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        # inititate the node with no neighbors at all at first
        self.neighbors = {UP:None, DOWN:None, LEFT:None, RIGHT:None, PORTAL:None}
        self.access = {UP:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       DOWN:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       LEFT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       RIGHT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT]}

    def denyAccess(self, direction, entity):
        if entity.name in self.access[direction]:
            self.access[direction].remove(entity.name)

    def allowAccess(self, direction, entity):
        if entity.name not in self.access[direction]:
            self.access[direction].append(entity.name)

    def render(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.position.asTuple()
                line_end = self.neighbors[n].position.asTuple()
                pygame.draw.line(screen, WHITE, line_start, line_end, 4)
                pygame.draw.circle(screen, RED, self.position.asInt(), 12)


class NodeGroup(object):
    def __init__(self, level):
        self.level = level # this is the txt file saved in the folder at the given level (initiated in the run.py script)
        self.nodesLUT = {}
        self.nodeSymbols = ['+', 'P', 'n']
        self.pathSymbols = ['.', '-', '|', 'p']
        data = self.readMazeFile(level) # read the data
        self.createNodeTable(data) # uses a lookup table to identify where the nodes are at in terms of coordinates
        # connect all the items in the level that are part of the nodeSymbols together in the form a dictonary
        # stating where every single node has neighbors on every direction (if it has no neighbors ion a said direction, then this shall be set to None as default)
        self.connectHorizontally(data)
        self.connectVertically(data)
        self.homekey = None
        self.costs = self.get_nodes()

    def readMazeFile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')

    # create a table where all the nodes are at (a look up tables)
    def createNodeTable(self, data, xoffset=0, yoffset=0):    
        # go through every single coordinate there is in the data and add the information to 
        # the lookup table as long as they are part of the nodeSymbol (['+', 'P', 'n'])
        for row in list(range(data.shape[0])):
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    x, y = self.constructKey(col+xoffset, row+yoffset) # adjust the coordinate to fit in the grid
                    self.nodesLUT[(x, y)] = Node(x, y)

    def constructKey(self, x, y): # adjusts the coordinates to be in adequate dimensions on the grid
        return x * TILEWIDTH, y * TILEHEIGHT


    def connectHorizontally(self, data, xoffset=0, yoffset=0):
        # go through every coordinate in the data. Connect all the nodes in the rows together
        for row in list(range(data.shape[0])):
            key = None # reinitate to None the key every time we are in a new column
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    if key is None:
                        # xoffset and yoffset are set to 0 as default (we do not really use it??)
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        # for every Node assign the neighbors to each other
                        self.nodesLUT[key].neighbors[RIGHT] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[LEFT] = self.nodesLUT[key]
                        key = otherkey
                # if the node is not part of the node symbols or path symbols then set the key to None (resetting it essentially)
                elif data[row][col] not in self.pathSymbols:
                    key = None

    def connectVertically(self, data, xoffset=0, yoffset=0):
        # this method does the same as connectHorizontally() but for getting the neighbors that
        # are on top or below of each other, not on the sides
        dataT = data.transpose()
        for col in list(range(dataT.shape[0])):
            key = None
            for row in list(range(dataT.shape[1])):
                if dataT[col][row] in self.nodeSymbols:
                    if key is None:
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        self.nodesLUT[key].neighbors[DOWN] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[UP] = self.nodesLUT[key]
                        key = otherkey
                elif dataT[col][row] not in self.pathSymbols:
                    key = None


    def getStartTempNode(self):
        nodes = list(self.nodesLUT.values())
        return nodes[0]

    # method that sets a portal pai. What is a portal??
    def setPortalPair(self, pair1, pair2):
        # each key is the coordinates adjusted to the grid's size
        key1 = self.constructKey(*pair1)
        key2 = self.constructKey(*pair2)
        # ensure that the keys are part of the nodes look up table (that that they are a nodeSymbol)
        if key1 in self.nodesLUT.keys() and key2 in self.nodesLUT.keys():
            # on the look up table, add to the two neighbors in question the coordinates of each other
            # I assume that these two nodes are then going to be able to transport each other's location
            self.nodesLUT[key1].neighbors[PORTAL] = self.nodesLUT[key2]
            self.nodesLUT[key2].neighbors[PORTAL] = self.nodesLUT[key1]

    def createHomeNodes(self, xoffset, yoffset):
        # use the homeoffset set in each level
        homedata = np.array([['X','X','+','X','X'],
                             ['X','X','.','X','X'],
                             ['+','X','.','X','+'],
                             ['+','.','+','.','+'],
                             ['+','X','X','X','+']])
        # create the nodes' lookup table (nodes are those with a symbol in ['+', 'P', 'n'])
        # connect all the nodes vertically and horizontally in the lookup table. 
        # But why use a fixed default data and not the level one????
        self.createNodeTable(homedata, xoffset, yoffset)
        self.connectHorizontally(homedata, xoffset, yoffset)
        self.connectVertically(homedata, xoffset, yoffset)
        self.homekey = self.constructKey(xoffset+2, yoffset) # the homkey will be based on the homeoffset for each maze. What is the point of this???
        return self.homekey

    def connectHomeNodes(self, homekey, otherkey, direction):    
        # add on the lookup table the predefined neighbors on the left and on the right to the homekey
        key = self.constructKey(*otherkey)
        self.nodesLUT[homekey].neighbors[direction] = self.nodesLUT[key]
        self.nodesLUT[key].neighbors[direction*-1] = self.nodesLUT[homekey]

    def getNodeFromPixels(self, xpixel, ypixel):
        if (xpixel, ypixel) in self.nodesLUT.keys():
            return self.nodesLUT[(xpixel, ypixel)]
        return None

    def getNodeFromTiles(self, col, row):
        x, y = self.constructKey(col, row)
        if (x, y) in self.nodesLUT.keys():
            return self.nodesLUT[(x, y)]
        return None

    def denyAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.denyAccess(direction, entity)

    def allowAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.allowAccess(direction, entity)

    def denyAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.denyAccess(col, row, direction, entity)

    def allowAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.allowAccess(col, row, direction, entity)

    def denyHomeAccess(self, entity):
        self.nodesLUT[self.homekey].denyAccess(DOWN, entity)

    def allowHomeAccess(self, entity):
        self.nodesLUT[self.homekey].allowAccess(DOWN, entity)

    def denyHomeAccessList(self, entities):
        for entity in entities:
            self.denyHomeAccess(entity)

    def allowHomeAccessList(self, entities):
        for entity in entities:
            self.allowHomeAccess(entity)

    def render(self, screen):
        for node in self.nodesLUT.values():
            node.render(screen)

    #############################
    # returns a list of all nodes in (x,y) format
    def getListOfNodesVector(self):        
        return list(self.nodesLUT)

    # returns a node in (x,y) format
    def getVectorFromLUTNode(self, node):        
        id = list(self.nodesLUT.values()).index(node)
        listOfVectors = self.getListOfNodesVector()
        return listOfVectors[id]

    # returns neighbors of a node in LUT form
    def getNeighborsObj(self, node):
        node_obj = self.getNodeFromPixels(node[0], node[1])
        return node_obj.neighbors

    # returns neighbors in (x,y) format
    def getNeighbors(self, node):
        neighs_LUT = self.getNeighborsObj(node)
        vals = neighs_LUT.values()
        neighs_LUT2 = []
        for direction in vals:
            if not direction is None:
                neighs_LUT2.append(direction)
        list_neighs = []
        for neigh in neighs_LUT2:
            list_neighs.append(self.getVectorFromLUTNode(neigh))
        return list_neighs

    # used to initialize node system for Dijkstra algorithm
    def get_nodes(self): # it returns where there are neighbors essentially (with a cost of 1, meaning how much it costs to walk on that edge)
        print('self.nodesLUT --> ',{k:v.position.__str__() for k,v in self.nodesLUT.items()})
        costs_dict = {}
        listOfNodesPixels = self.getListOfNodesVector()
        for node in listOfNodesPixels:
            neigh = self.getNeighborsObj(node)
            temp_neighs = neigh.values()
            temp_list = []            
            for direction in temp_neighs:
                if not direction is None:
                    temp_list.append(1)
                else:
                    temp_list.append(None)
            costs_dict[node] = temp_list            
        return costs_dict
    


# [(16, 64), (96, 64), (192, 64), (240, 64), (336, 64), (416, 64), (16, 128), (96, 128), (144, 128), (192, 128), (240, 128), (288, 128), (336, 128), (416, 128), (16, 176), (96, 176), (144, 176), (192, 176), (240, 176), (288, 176), (336, 176), (416, 176), (144, 224), (192, 224), (240, 224), (288, 224), (0, 272), (96, 272), (144, 272), (288, 272), (336, 272), (432, 272), (144, 320), (288, 320), (16, 368), (96, 368), (144, 368), (192, 368), (240, 368), (288, 368), (336, 368), (416, 368), (16, 416), (48, 416), (96, 416), (144, 416), (192, 416), (240, 416), (288, 416), (336, 416), (384, 416), (416, 416), (16, 464), (48, 464), (96, 464), (144, 464), (192, 464), (240, 464), (288, 464), (336, 464), (384, 464), (416, 464), (16, 512), (192, 512), (240, 512), (416, 512)]    