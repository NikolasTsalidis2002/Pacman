import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData


# the plan is to make Pacman follow a fell behaviour, and predict where it wants to go.
# this predcition will be based on following a* attempting to get there

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives) # it will have the number of lives specified above (5 in this case)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData() # initate the two mazes found in the mazedata.py 

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level) # initiate the maze depending on the level we are at (at first the first one)
        # open the files that we want to use as the grid for the level (there are two per level -> normal and rotate file)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        # initiate the NodeGroup class giving it the level text file (normal one) as the level argument
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt") # at this point we have a lookup table used to identify all the neighbors of all the nodes
        self.mazedata.obj.setPortalPairs(self.nodes) # to the nodes in question (given at each Maze intiatilization), add to their portal the coordinate of the other pair's node in the lookup table
        self.mazedata.obj.connectHomeNodes(self.nodes) # i beleive believe this homkey is a default combination of nodes that all levels must have (that way we ensure all levels start the same way)        
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt") # initiate the PelletGroup giving it the level file as an argument. At this point one should have a list of all pellets and all powerPellets with its own list as well              
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart), pellets=self.pellets, nodes=self.nodes) # initiate Pacman        
        # intiate all the ghosts. At first all the nodes are going to start as default on the same node (where the first level's node)
        # update the start node for every single one of the ghosts. 
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)       
        self.pacman.getGhostObject(self.ghosts)        
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman) # do not allow Pacman to go down when it has reached home
        self.nodes.denyHomeAccessList(self.ghosts) # do not allow ghosts to go down when reached home
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky) # do the same but for right to inky
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde) # and the same for clyde to the left
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes) # do the same for ghosts at all nodes   

    def startGame_old(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

        

    def update(self):
        # update the texts, pellets, and if not paused, ghosts
        dt = self.clock.tick(60) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents() # check th pellets (if there are no more pellets then the level is won)
            self.checkGhostEvents() # check the ghost events
            self.checkFruitEvents() # check the fruits

        # if it has not been eaten in the checkGhostEvents and it is not paused,
        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if self.flashBG: # if there are no more pellets
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime: # if the timer is greater than default falsh time (0.2)
                self.flashTimer = 0
                # make background falsh (visual effect)
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm
        # update the pause. Make the function take place (win, lose, ate fruit or ghost)
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod() # execute whatever happens after the game is paused
        self.checkEvents() # check if we are quitting of the space key board is pressed
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        # keep track of all the pellets eaten
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                # if Pacman has eaten all the pellets, then hide entities, pause game and 
                # give nextLevel as function for the pauser class to use in the next update
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost): # if Pacman has collided with a ghost
                if ghost.mode.current is FREIGHT: # if mode is Freight (after power pellet)
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points) # add points to the score based on what Pacman ate (every entity has a different score)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    # if Pacman eats a ghost in freight mode then the number of points per ghost increases by a factor of 2
                    self.ghosts.updatePoints()
                    # pause for a second and then show all the entities again
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    # when eaten, make that ghost start spawning again and allow
                    # that ghost to be able to go down in the homekey coordinate
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                # if pacman eaten in a state which is not spawn (chase or scatter)
                elif ghost.mode.current is not SPAWN:
                    # remove a life, remove the image, make pacman die (making his direction to be stop and setting alive to False)
                    # hide the ghosts. Depending on the number of lives, restart the game or the level
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                # initiate the fruit to be in a fixed tile at a given level
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                print(self.fruit)
        # if there is a fruit
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit): # if pacman is eating eat
                self.updateScore(self.fruit.points) # add the fruit points to the score
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image) # keep track of the eaten fruits
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities() # show the entities again
        self.level += 1 # increase the level up by 1
        self.pause.paused = True # set to puase as default
        self.startGame() # start the game by creating the grid, with all connections and so on
        self.textgroup.updateLevel(self.level) # update level number in the text

    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()



