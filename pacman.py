import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
import numpy as np
from random import choice
from algorithms import dijkstra, dijkstra_or_a_star

class Pacman(Entity):
    # it inherits form Entity
    def __init__(self, node, pellets, nodes):
        Entity.__init__(self, node )      
        self.pellets = pellets     
        self.nodes = nodes
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT # by default it is going to start looking towards the left          
        print(self.node.neighbors[LEFT])  
        self.setBetweenNodes(LEFT)      
        self.alive = True
        self.sprites = PacmanSprites(self)
        self.directionMethod = self.goalDirection

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    # find all the pellets that there are in the game
    def closest_pellet(self):
        # get all the pellets there are
        pellets = {k:v for k,v in dict(enumerate(self.pellets.pelletList)).items() if v.name==POWERPELLET}
        pellets = dict(enumerate(list(pellets.values())))    
        if len(pellets) != 0:
            distances_pellet_pacman = []
            for k,pellet in pellets.items():
                goal = pellet.position
                distance = self.position-goal            
                distances_pellet_pacman.append(distance.magnitudeSquared())
            closest_pellet = np.argmin(distances_pellet_pacman)        
            closest_pellet = pellets[closest_pellet]
            # self.goal = closest_ghost.position        
            return closest_pellet
        else:
            return False
    
    #############
    # NEW -> THIS IS NOT PART OF THE GAME
    # Executes Dijkstra from Ghost's previous node as start 
    # to pacman's target node as target.
    def getDijkstraPath(self,target,current_node):
        start_node = current_node
        print('\tstarting node --> ',start_node.position)
        print('\ttarget node --> ',target.position)
        start_node = self.nodes.getVectorFromLUTNode(start_node)
        # previous_nodes, shortest_path = dijkstra(self.nodes, ghostTarget)
        previous_nodes, shortest_path = dijkstra_or_a_star(nodes=self.nodes, start_node=start_node, a_star=True)
        print('\tprevious_nodes --> ',previous_nodes)
        # print('These are the previouse nodes --> {}'.format(previous_nodes))
        starting_node_pos = start_node
        target_node_pos = (target.position.x,target.position.y)
        node = target_node_pos
        path = [node]
        while node != starting_node_pos:
            if node in previous_nodes:
                node = previous_nodes[node]
                path.append(node)
            else:
                return False
        
        path.reverse()        
        return path

    # NEW -> THIS IS NOT PART OF THE GAME
    # Chooses direction in which to turn based on the dijkstra
    # returned path
    def goalDirectionDij(self,target,current_node):
        directions = self.directions
        path = self.getDijkstraPath(target,current_node)
        if not path:
            return False
        print('\tPath to follow -> ',path)
        next_pos = path[1]
        current_pos = path[0]
        print('\tcurrent_pos vs next_pos',current_pos,next_pos)
        if current_pos[0] > next_pos[0] and 2 in directions : #left
            return 2
        if current_pos[0] < next_pos[0] and -2 in directions : #right
            return -2
        if current_pos[1] > next_pos[1] and 1 in directions : #up
            return 1
        if current_pos[1] < next_pos[1] and -1 in directions : #down
            return -1
        else:
            if -1 * self.pacman.direction in directions:
                return -1 * self.pacman.direction
            else: 
                return choice(directions)

    # NEW -> THIS IS NOT PART OF THE GAME
    def getGhostObject(self, ghosts):
        self.ghosts = {}
        counter = 0
        for ghost in ghosts:
            self.ghosts[counter] = ghost
            counter += 1

    # NEW -> THIS IS NOT PART OF THE GAME
    def goalDirection(self,directions,mode='fleeing'): # maximize the distance between the pacman and the ghost
        distances = []
        for direction in directions:
            vec = self.node.position  + self.directions[direction]*TILEWIDTH - self.goal
            distances.append(vec.magnitudeSquared())
        if mode == 'chasing':
            index = distances.index(min(distances))
        else:
            index = distances.index(max(distances))        
        return directions[index]


    def choose_direction(self,directions):
        # find the ghost that is the closest to pacman
        distances_ghost_pacman = []
        for k,ghost in self.ghosts.items():
            goal = ghost.position
            distance = self.position-goal            
            distances_ghost_pacman.append(distance.magnitudeSquared())
        closest_ghost = np.argmin(distances_ghost_pacman)        
        closest_ghost = self.ghosts[closest_ghost]
        self.ghost = closest_ghost # this is to keep track of the ghosts. If they FREIGHT we want to be able to eat them even before we get to out Target
        # given on the closest ghosts mode, then we can chase pellets
        self.closest_ghost = closest_ghost
        self.modes = {0:'scatter',1:'chase',2:'freight',3:'spawn'}
        closest_ghost_mode = closest_ghost.mode.current #self.ghost.mode.current
        print('running away from: ',closest_ghost,self.modes[closest_ghost_mode])  
        
        if closest_ghost_mode in [0,3]: # if closest ghost is scatter, then chase the pellet
            current_node = self.node
            closest_pellet = self.closest_pellet()
            if not closest_pellet:
                print('We are since there are no more pellets to collect')
                self.goal = closest_ghost.position
                direction = self.goalDirection(directions)

            else: 
                for node in self.nodes.nodesLUT.values():
                    if closest_pellet.position == node.position:
                        closest_pellet = node
                self.goal = closest_pellet.position
                direction = self.goalDirectionDij(closest_pellet,current_node)
                print('\tThis is the direction we have decided to take: ',direction)
                
                if not direction:                
                    print('We are fleeing from the ghost since the direction was false')
                    self.goal = closest_ghost.position
                    direction = self.goalDirection(directions)                                

        elif closest_ghost_mode == 1: # if the closest ghost is chasing, then run
            print('We are fleeing from the ghost')
            self.goal = closest_ghost.position
            direction = self.goalDirection(directions)
        elif closest_ghost_mode == 2: # if the closest ghost is freight, then chase it
            print('We are chasing the ghost')
            self.goal = closest_ghost.position
            direction = self.goalDirection(directions,mode='chasing')
         
        return direction

        

    def update(self, dt):
        # check the mode of the ghosts
        try:
            print('Ghosts current mode --> ',self.modes[self.ghost.mode.current])
        except: pass
        
        self.sprites.update(dt)
        # calculate the new position based on the direction of the previous iteration
        self.position += self.directions[self.direction]*self.speed*dt
        # direction = self.getValidKey() # check that we have pressed a valid keyboard
        direction = self.direction # check that we have pressed a valid keyboard
        print('position vs target --> ',self.position.__str__(),self.target.position.__str__())
        if self.overshotTarget(): # if Pacman has gotten to its target
            print('we have overshot the target __________________####')
            self.node = self.target # update the node we are keeping track of
            directions = self.validDirections()
            direction = self.choose_direction(directions) # this functino is for determing the best direction to take
            print('\tThis is the direction we have decided to take --> ',direction)
            # in the case the node in question is a portal, transport to the other end of the portal
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction) # get the new target (vector to go to)
            # check if the target is not the same as the as the node Pacman is currently at

            if self.target is not self.node:
                # if not the same, then make the class direction the direction caused by the key
                self.direction = direction
            else:
                print('\tThe target we had in mind is the same as the current node')
                # if target is same as node, make target be the node that comes after the current position on the same direction (essentially keep going)
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node: # make the direction stop
                self.direction = STOP
            self.setPosition()
        else: # if it has not gotten to the target keep going. Unless, the user has reversed the direction
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        l = [key_pressed[K_UP],key_pressed[K_DOWN],key_pressed[K_LEFT],key_pressed[K_RIGHT]]
        if True in l:
            print(key_pressed[K_UP],key_pressed[K_DOWN],key_pressed[K_LEFT],key_pressed[K_RIGHT])
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP  

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        # check if we are colliding (the distance between the enities is less than the determined collision radius
        # where the pacman is, and where it is trying to get to (ie: fruit)
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
