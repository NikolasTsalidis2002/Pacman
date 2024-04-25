import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import datetime

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
        self.total_episodes = 1000

        self.paused_check = False        
        self.reset_level_check = False

        self.save_qtable = False

    def initiate_game(self):
        self.startGame()
        self.pellets_numEaten = 0



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
        
        print('\n\tself.pacman --> ',self.pacman_position)     
        print('\tself.ghosts --> ',self.ghosts_positions)
        print('\tself.closest_pellets --> ',closest_pellets)        

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
        
        return tuple(state)


    # Assume we have a function that can get all possible actions for a given state
    def getLegalActions(self):
        # Returns all legal actions for a given state
        directions_valid = []
        for i in [UP, DOWN, LEFT, RIGHT]:
            if self.pacman.validDirection(i):
                directions_valid.append(i)
        self.directions_valid = directions_valid
        print('\t\tdirections_valid --> ',directions_valid)

        return self.directions_valid
    

    def update_iteration(self):
        print('\n\t#### we are updating the game')
        # dt = self.clock.tick(10)
        dt = self.clock.tick(30) / 1000.0
        dt = self.clock.tick(30) / 300.0
        self.pellets.update(dt)

         # unpause the game in the case that it is paused and remove the text
        self.checkEvents()
        if self.pause.paused: 
            if not self.paused_check:
                self.pause.flip()                
                self.showEntities()
        
        if self.paused_check: 
            print('we are in qlearning')
            print('self.pause.paused --> ',self.pause.paused)
            [][0]

        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

            if self.reset_level_check:
                print('self.reset_level_check from qlearning.py --> ',self.reset_level_check)
            
        # check if a pellet has been eaten
        if self.pellets_numEaten != self.pellets.numEaten:
            self.pellets_numEaten += 1
            self.atePellet = True
        else: self.atePellet = False        

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
        
        self.render()               
                

    def getReward(self):
        # print('self.atePellet --> ',self.atePellet)
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
        exploration_start,exploration_end = 1.0,0.01
        actions_random_ratio_list,iterations_per_episode_list = [],[]

        # Training loop
        for episode in range(self.total_episodes):
            self.restart_game_check = False # reset this every time a new episode starts
            print(f'\n{episode+1}/{self.total_episodes}')

            state = self.getStateRepresentation()
            valid_actions = self.getLegalActions()

            if state not in Q_table:                    
                Q_table[state] = {act: 0 for act in valid_actions} 

            if not self.pacman.alive:
                print('\tPacMan is dead. Restarting the game!')
                print('\tself.lives --> ',self.lives)
                
            # while pacman is alive, train the model. If dies, reset level and proceed. Do this until all episodes are done
            iteration_number = 0
            while self.pacman.alive:                
                self.textgroup.alltext[READYTXT].visible = False     # hide the text
                if self.restart_game_check: break                    # if we are starting a new game, break the loop and go to the next episode                

                print(f'\n\nNew iteration {iteration_number} in episode {episode}\n\tstate --> ',state)

                # find the action to tkae
                rand = np.random.rand()                    
                # this function gives us a rate that is to be used to determine if we do a random action or not (given that the prob is higher or below it)
                # the closer it is to the end episode, the more likely it will not do a random action
                exploration_rate = exploration_start * (exploration_end / exploration_start) ** (episode / self.total_episodes) 
                # get the valid actions at the present time                    

                if rand < exploration_rate or state not in Q_table:                       
                    print('\t\tWe are getting a random action')
                    self.action = random.choice(valid_actions)
                    actions_random_ratio_list.append('random')

                else:
                    print('\t\twe are getting the best action')
                    self.action = max(Q_table[state], key=Q_table[state].get)
                    actions_random_ratio_list.append('best')

                print('\t\t#####self.action --> ',self.action)
                # Take action, observe new state and reward
                self.update_iteration()
                print('self.reset_level_check after the update function --> ',self.reset_level_check)
                if self.reset_level_check:
                    iterations_per_episode_list.append(iteration_number)
                    print('iterations_per_episode_list --> ',iterations_per_episode_list)
                    # [][0]
                    # reset the info for the new life round
                    self.reset_level_check = False
                    iteration_number = 0

                newState = self.getStateRepresentation()
                print('\t\tnewState --> ',newState)
                reward = self.getReward() # i have no idea how to do this one
                # update the Q table witht the reward to the action
                
                print('\t\treward --> ',reward)
                # Q-learning update
                if newState not in Q_table:                                            
                    Q_table[newState] = {act: 0 for act in self.getLegalActions()}

                # update the Q table using the QLearning formula
                try:
                    new_reward = (1 - self.alpha) * Q_table[state][self.action] + \
                                            self.alpha * (reward + self.gamma * max(Q_table[newState].values()))
                except:
                    print('Q_table[state] --> ',Q_table[state])   
                print(f'\t\tAction: {self.action}\tReward: {new_reward}')
                Q_table[state][self.action] = new_reward

                print('\t\tQ_table')
                counter = 0
                for k,v in Q_table.items(): 
                    if counter > len(Q_table)-5:
                        print(f'\t\t\t{k} --> ',v)
                    counter += 1

                # Transition to the new state
                state = newState
                iteration_number += 1

                print('iterations_per_episode_hist --> ',iterations_per_episode_list)
                
            
            if episode%2 == 0:
                # First plot for viewing the action randomness ditribution
                plt.figure(1)  # Create a new figure
                plt.hist(actions_random_ratio_list)
                plt.title(episode)
                plt.show(block=False)
                plt.pause(0.001)  # Pause for a brief moment to ensure the plot is rendered
                
                # Second plot for viewing the durection of the lives (the more the better)
                plt.figure(2)  # Create a new figure
                plt.plot(iterations_per_episode_list)
                plt.title('Iterations per episode')
                plt.show(block=False)
                plt.pause(0.001)  # Pause for a brief moment to ensure the plot is rendered

            if self.save_qtable:
                if episode%10 == 0 and episode != 0:
                    # save the Q table as a pickle file
                    current_time = str(datetime.datetime.now())
                    with open(f'Qtables/Qtable_{current_time}.pkl', 'wb') as f:
                        pickle.dump(Q_table, f)                      
                                    
        # Run the game with the learned policy
        self.Q_table = Q_table
        
        if self.save_qtable:
            # save the Q table as a pickle file
            current_time = str(datetime.datetime.now())
            with open(f'Qtables/Qtable_{current_time}.pkl', 'wb') as f:
                pickle.dump(self.Q_table, f)        



if __name__ == "__main__":
    ql = QLearning()
    # ql.getStateRepresentation()
    ql.initiate_game()
    ql.train()
    print('\n\n',ql.Q_table.values())
    