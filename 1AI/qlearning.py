import numpy as np
import random

# import pygame
# from pygame.locals import *
from constants import *
# from pacman import Pacman
# from nodes import NodeGroup
# from pellets import PelletGroup
# from ghosts import GhostGroup
# from fruit import Fruit
# from pauser import Pause
# from text import TextGroup
# from sprites import LifeSprites
# from sprites import MazeSprites
# from mazedata import MazeData
from run import GameController


class QLearning(GameController):
    # def __init__(self,pacman:Pacman,pellets:PelletGroup,ghosts:GhostGroup):
    def __init__(self):
        super().__init__()
        # self.pacman = pacman
        # self.pellets = pellets
        # self.ghosts = ghosts

        # Q-learning parameters
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.9  # exploration rate
        self.total_episodes = 5


    def initiate_game(self):
        self.startGame()
        self.pellets_numEaten = 0
        print('self.pacman.alive --> ',self.pacman.alive)



    # Assume we have a function that can convert a game state to a simplified state representation
    def getStateRepresentation(self):
        state = []
        # Convert gameState to a state tuple        
        # For example: (Pacman's x position, Pacman's y position, ghost1's x position, ghost1's y position, ...)      
        self.pacman_position = self.pacman.position.asTuple()
        self.pellets_positions = [i.position.asTuple() for i in self.pellets.pelletList]
        self.powerpellets_positions = [i.position.asTuple() for i in self.pellets.powerpellets]
        self.ghosts_positions = [i.position.asTuple() for i in self.ghosts.ghosts]
        # find only only the closest pellets
        pac_pell_dists = [tuple(np.abs(np.array(i)-np.array(self.pacman_position))) for i in self.pellets_positions+self.powerpellets_positions]
        pac_pell_dists = {i:np.sqrt(i[0]**2+i[1]**2) for i in pac_pell_dists}
        pac_pell_dists = {k:v for k,v in sorted(pac_pell_dists.items(),key=lambda i:i[1])}

        top_n_pellets = 5
        if len(pac_pell_dists) > top_n_pellets: closest_pellets = list(pac_pell_dists.keys())[:top_n_pellets]
        else: closest_pellets = pac_pell_dists
        
        print('\nself.pacman --> ',self.pacman_position)     
        print('self.ghosts --> ',self.ghosts_positions)
        print('self.closest_pellets --> ',closest_pellets)        

        state.append(self.pacman_position)

        # Ghosts' positions, ensure a fixed length for consistency
        for i in range(len(self.ghosts_positions)):
            if i < 4:
                state.append(self.ghosts_positions[i])
            else:
                state.append(np.nan)  # Use a placeholder if there are fewer ghosts

        # Closest pellets' positions, also ensure a fixed length
        for i in range(len(closest_pellets)):
            if i < top_n_pellets:
                state.append(closest_pellets[i])
            else:
                state.append((np.nan,np.nan))  # Use a placeholder if there are fewer pellets
        print('state --> ',tuple(state))
        return tuple(state)


    # Assume we have a function that can get all possible actions for a given state
    def getLegalActions(self):
        # Returns all legal actions for a given state
        directions_valid = []
        for i in [UP, DOWN, LEFT, RIGHT]:
            if self.pacman.validDirection(i):
                directions_valid.append(i)
        self.directions_valid = directions_valid
        print('directions_valid --> ',directions_valid)

        return self.directions_valid
    

    def update_iteration(self):
        print('#### we are updating the game')
        # dt = self.clock.tick(10)
        # dt = self.clock.tick(30) / 1000.0
        dt = self.clock.tick(30) / 200.0
        self.pellets.update(dt)

        if self.pause.paused: self.pause.flip() # unpause the game in the case that it is paused
        print('self.pause.paused --> ',self.pause.paused)

        self.ghosts.update(dt)      
        if self.fruit is not None:
            self.fruit.update(dt)
        self.checkPelletEvents()
        self.checkGhostEvents()
        self.checkFruitEvents()
        
        # check if a pellet has been eaten
        if self.pellets_numEaten != self.pellets.numEaten:
            self.pellets_numEaten += 1
            self.atePellet = True
        else: self.atePellet = False
    
        # print('we are still pasued!!! ')
            

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt,action=self.action)        
        else:
            self.pacman.update(dt,action=self.action)     

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()               
                

    def getReward(self):
        reward = 0
        # Check if Pacman ate a pellet
        if self.atePellet: reward += 10  # Reward for eating a pellet
        # Check if Pacman ate a ghost
        if self.ateGhost: reward += 200  # Reward for eating a ghost
        # Check if Pacman died
        if not self.pacman.alive: reward -= 500  # Penalty for dying
        return reward



    def train(self):
        # Initialize Q-table, this will need to be adjusted for your state space and action space
        Q_table = {}

        # Training loop
        for episode in range(self.total_episodes):
            state = self.getStateRepresentation()
            if state not in Q_table:                    
                Q_table[state] = {act: 0 for act in self.getLegalActions()}            
            print('\nstate --> ',state)

            if self.pacman.alive:
                while self.pacman.alive:
                    print('\n')
                    rand = np.random.rand()
                    if rand < self.epsilon or state not in Q_table:
                        print('We are getting a random action')
                        self.action = random.choice(self.getLegalActions())
                    else:
                        print('we are getting the best action')
                        self.action = max(Q_table[state], key=Q_table[state].get)
                    print('\t#####33self.action --> ',self.action)
                    # Take action, observe new state and reward
                    self.update_iteration()
                    newState = self.getStateRepresentation()
                    print('\tnewState --> ',newState)
                    reward = self.getReward() # i have no idea how to do this one
                    
                    # Q-learning update
                    if newState not in Q_table:                    
                        Q_table[newState] = {act: 0 for act in self.getLegalActions()}
                        print('\tQ_table')
                        counter = 0
                        for k,v in Q_table.items(): 
                            if counter > len(Q_table)-5:
                                print(f'\t\t{k} --> ',v)
                            counter += 1

                    Q_table[state][self.action] = (1 - self.alpha) * Q_table[state][self.action] + \
                                            self.alpha * (reward + self.gamma * max(Q_table[newState].values()))

                    # Transition to the new state
                    state = newState
            else: print('PacMan is dead')
        # Run the game with the learned policy



if __name__ == "__main__":
    ql = QLearning()
    # ql.getStateRepresentation()
    ql.initiate_game()
    ql.train()
    