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
    def __init__(self, node, pellets, nodes, pacman_comments=False, not_qlearning=True):
        Entity.__init__(self, node )      
        self.pellets = pellets     
        self.nodes = nodes
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT # by default it is going to start looking towards the left          
        self.setBetweenNodes(LEFT)      
        self.alive = True
        self.sprites = PacmanSprites(self)
        self.directionMethod = self.goalDirection
        self.pacman_comments = pacman_comments
        self.not_qlearning = not_qlearning
        self.won_game_check = False

        self.overshot_check = False

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
        pellets = dict(enumerate(list(pellets.values()))) # reenumerate the dict for power pellets only
        path_pellets = {k:v for k,v in dict(enumerate(self.pellets.pelletList)).items() if v.name!=POWERPELLET}
        path_pellets = dict(enumerate(list(path_pellets.values()))) # reenumerate the dict for path pellets only
        
        distances_path_pellet_pacman = []
        for k,pellet in path_pellets.items():
            goal = pellet.position
            distance = self.position-goal            
            distances_path_pellet_pacman.append(distance.magnitudeSquared())

        # from all the closest path pellets there are, chase that one which is furthest away from the ghost  
        if len(distances_path_pellet_pacman) == 0: 
            print('WE WON THE GAME!')
            self.won_game_check = True
            return False
        closest_path_pellets = np.where(np.array(distances_path_pellet_pacman) == min(distances_path_pellet_pacman))[0]
        closest_path_pellets = [path_pellets[i] for i in closest_path_pellets]
        path_pellets_ghost_distances = [(self.ghost.position-i.position).magnitudeSquared() for i in closest_path_pellets]
        closest_path_pellet = closest_path_pellets[np.argmax(path_pellets_ghost_distances)]

        # ensure that there are path pellets left. Otherwise return False
        if len(path_pellets) != 0:
            distances_pellet_pacman = []
            for k,pellet in pellets.items():
                goal = pellet.position
                distance = self.position-goal            
                distances_pellet_pacman.append(distance.magnitudeSquared())
            if len(distances_pellet_pacman) != 0:
                closest_pellet = np.argmin(distances_pellet_pacman)        
                closest_pellet = pellets[closest_pellet]
                # self.goal = closest_ghost.position   
                if self.pacman_comments: 
                    print('\t### Closest path pellet --> ',closest_path_pellet,'at distance',distances_path_pellet_pacman[np.argmin(distances_path_pellet_pacman)])
                    print('\t### Closest pellet --> ',closest_pellet,'at distance',distances_pellet_pacman[np.argmin(distances_pellet_pacman)])
            else:                
                closest_pellet = None            
                if self.pacman_comments: 
                    print('\t### Closest path pellet --> ',closest_path_pellet,'at distance',distances_path_pellet_pacman[np.argmin(distances_path_pellet_pacman)])
                    print('\t### Closest pellet --> ',closest_pellet,'at distance',closest_pellet) 
            # store these as instance attributes
            self.closest_path_pellet = closest_path_pellet
            self.closest_power_pellet = closest_pellet    
            try: # if it failes that means that there are no more power pellets, so we have to chase path pellets
                ratio = distances_path_pellet_pacman[np.argmin(distances_path_pellet_pacman)]/distances_pellet_pacman[np.argmin(distances_pellet_pacman)]
            except:
                ratio = 0.01
            if self.pacman_comments: print('\t### ratio --> ',ratio)
            if ratio < 0.3:
                if self.pacman_comments:  print('\t\tStill far away from power pellet. Chasing path pellet')
                return ('path_pellet',closest_path_pellet)
            else:
                if self.pacman_comments: print('\t\tCloser to power pellet. Chasing power pellet')
                return ('power_pellet',closest_pellet)
        else:
            return False
    
    #############
    # NEW -> THIS IS NOT PART OF THE GAME
    # Executes Dijkstra from Ghost's previous node as start 
    # to pacman's target node as target.
    def getDijkstraPath(self,target,current_node):
        start_node = current_node
        if self.pacman_comments: 
            print('\tstarting node --> ',start_node.position)
            print('\ttarget node --> ',target.position)
        start_node = self.nodes.getVectorFromLUTNode(start_node)
        # previous_nodes, shortest_path = dijkstra(self.nodes, ghostTarget)
        previous_nodes, shortest_path = dijkstra_or_a_star(nodes=self.nodes, start_node=start_node, a_star=True)
        if self.pacman_comments:  print('\tprevious_nodes --> ',previous_nodes)
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
        if self.pacman_comments: print('\tPath to follow -> ',path)
        next_pos = path[1]
        current_pos = path[0]
        if self.pacman_comments: print('\tcurrent_pos vs next_pos',current_pos,next_pos)
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
            if self.pacman_comments: print('\t\t\t???',directions,'These are the distances we can choose from when Chasing --> ',distances)
            index = distances.index(min(distances))
        else:
            if self.pacman_comments: print('\t\t\t???',directions,'These are the distances we can choose from when fleeing --> ',distances)
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
        self.modes = {0:'scatter',1:'chase',2:'freight',3:'spawn'}
        closest_ghost_mode = closest_ghost.mode.current #self.ghost.mode.current
        if self.pacman_comments: print('running away from: ',closest_ghost,self.modes[closest_ghost_mode])  
        
        closest_pellet = self.closest_pellet()
        # if we won the game then stay still
        if self.won_game_check: return 0

        if not closest_pellet: # ensure that there are pellets left. Otherwise flee from ghost
            if self.pacman_comments: print('We are running  since there are no more pellets to collect')
            self.goal = closest_ghost.position
            direction = self.goalDirection(directions,mode='fleeing')
            self.status = 'Fleeing from ghost'

        elif closest_ghost_mode in [0,3]: # if closest ghost is scatter, then chase the pellet
            current_node = self.node
            # check if we are to go after a power pellet or not
            if closest_pellet[0] == 'power_pellet':
                closest_pellet = closest_pellet[1]
                for node in self.nodes.nodesLUT.values():
                    if closest_pellet.position == node.position:
                        closest_pellet = node
                self.goal = closest_pellet.position
                direction = self.goalDirectionDij(closest_pellet,current_node)
                self.status = 'Chasing pellet'
                
                if not direction:    
                    if self.pacman_comments: print('Could not find shortest path for pellet')
                    if closest_ghost_mode == 3:      
                        if self.pacman_comments: print('\tSo we are going to chase the ghost')
                        self.goal = closest_ghost.position
                        direction = self.goalDirection(directions,mode='chasing')
                        self.status = 'Chasing ghost'                        
                    else:
                        if self.pacman_comments: print('\tSo we are going to flee from the ghost')
                        self.goal = closest_ghost.position
                        direction = self.goalDirection(directions,mode='fleeing') 
                        self.status = 'Fleeing from ghost'   
            else:
                closest_pellet = closest_pellet[1]
                self.goal = closest_pellet.position
                direction = self.goalDirection(directions,mode='chasing') 
                self.status = 'Chasing path pellet'                   

        elif closest_ghost_mode == 1: # if the closest ghost is chasing, then run
            self.goal = closest_ghost.position
            path_pellet_distance = (self.closest_path_pellet.position-self.position).magnitudeSquared()
            ghost_distance = (self.goal-self.position).magnitudeSquared()
            if self.pacman_comments: print('\t???power_pellet_distance vs ghost_distance',path_pellet_distance,ghost_distance,ghost_distance/path_pellet_distance)
            ratio = ghost_distance/path_pellet_distance
            # ratio of 2 won level one
            if ratio > 4: # if the ghost is chasing us, but we are still far enough, the follow the pellets
                if self.pacman_comments: print('\t\t###### We are being chased but we are sitll far enough! Chase pellet')
                self.goal = self.closest_path_pellet.position
                direction = self.goalDirection(directions,'chasing')
                self.status = 'Chasing pellet, ghost still not close.'                
            else:
                if self.pacman_comments: print('\t\t###### We are being chased and ghost too close. Fleee!')
                direction = self.goalDirection(directions,mode='fleeing')
                self.status = 'Fleeing from ghost'
        elif closest_ghost_mode == 2: # if the closest ghost is freight, then chase it
            self.goal = closest_ghost.position
            # print('???? The closest ghost is freight at a position --> ',self.goal)
            direction = self.goalDirection(directions,mode='chasing')
            self.status = 'Chasing ghost'
         
        return direction



    def update(self, dt, desired_action=None):        
        self.sprites.update(dt)
        # calculate the new position based on the direction of the previous iteration
        if self.not_qlearning:
            self.position += self.directions[self.direction]*self.speed*dt
        # direction = self.getValidKey() # check that we have pressed a valid keyboard
        direction = self.direction # check that we have pressed a valid keyboard

        # check the mode of the ghosts
        try:
            if self.pacman_comments: print('position vs target --> ',self.position.__str__(),self.target.position.__str__(),'\t\t||\tGhosts current mode --> ',self.ghost,self.modes[self.ghost.mode.current],'\tPacman status --> ',self.status)        
        except: 
            if self.pacman_comments: print('position vs target --> ',self.position.__str__(),self.target.position.__str__(),'\t#### failling to print status')
        
        if self.overshotTarget(): # if Pacman has gotten to its target
            self.overshot_check = True
            if self.pacman_comments: print('we have overshot the target __________________####')
            self.node = self.target # update the node we are keeping track of
            directions = self.validDirections()
            self.pacman_directions = directions

            if desired_action is not None: direction = desired_action
            else: direction = self.choose_direction(directions) # this functino is for determing the best direction to take

            if self.pacman_comments: print('\tThis is the direction we have decided to take --> ',direction)
            # in the case the node in question is a portal, transport to the other end of the portal
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            try: self.target = self.getNewTarget(direction) # get the new target (vector to go to)
            except: self.target = self.node
            # check if the target is not the same as the as the node Pacman is currently at

            if self.target is not self.node:
                # if not the same, then make the class direction the direction caused by the key
                self.direction = direction
            else:
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
