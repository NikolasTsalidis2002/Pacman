import logging
import os
import pickle
from filelock import FileLock
import numpy as np
from run import GameController
import matplotlib.pyplot as plt
import datetime

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os
import datetime

from constants import *
from run import GameController


import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os
import datetime
from collections import Counter



# def setup_logging():
#     # Create a custom logger
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)

#     # Create handlers
#     c_handler = logging.StreamHandler()  # Console handler
#     f_handler = logging.FileHandler('training.log')  # File handler

#     # Create formatters and add it to handlers
#     c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     c_handler.setFormatter(c_format)
#     f_handler.setFormatter(f_format)

#     # Add handlers to the logger
#     logger.addHandler(c_handler)
#     logger.addHandler(f_handler)

#     return logger


class QLearning(GameController):
    def __init__(self, save_qtable=True, use_last_saved_q_table=True, plot_performance=False, comments=False):
        super().__init__()

        # Q-learning parameters
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        # self.epsilon = 0.9  # exploration rate
        self.total_episodes = 100

        # do not touch these!
        self.paused_check = False        
        self.reset_level_check = False

        # these are just the arguments. Change the arguments, not the variables
        self.save_qtable = save_qtable
        self.use_last_saved_q_table = use_last_saved_q_table
        self.plot_performance = plot_performance
        self.comments = comments

        # self.qtable_filename = 'Qtables/shared_q_table.pkl'  # Set a common filename for the Q-table
        # self.logger = setup_logging()  # Initialize and assign the logger before any logging operation
        # self.logger.info("QLearning instance created, logger is now active.")

 
    # def load_q_table(self):
    #     filename = self.qtable_filename
    #     lock = FileLock(f"{filename}.lock")
    #     with lock:
    #         if os.path.exists(filename) and os.path.getsize(filename) > 0:  # Check if file exists and is not empty
    #             try:
    #                 with open(filename, 'rb') as file:
    #                     q_table = pickle.load(file)
    #                 self.logger.info(f"Loaded Q-table from {filename}.")
    #             except Exception as e:
    #                 self.logger.error(f"Failed to load Q-table from {filename}: {e}")
    #                 q_table = {}  # Initialize Q-table if file is corrupted
    #         else:
    #             q_table = {}  # Initialize Q-table if file does not exist or is empty
    #             self.logger.info("No Q-table found or file is empty. Initializing new Q-table.")
    #     return q_table
    
    
    # def save_q_table(self, current_q_table):
    #     filename = self.qtable_filename
    #     lock = FileLock(f"{filename}.lock")
    #     with lock:
    #         if os.path.exists(filename):
    #             with open(filename, 'rb') as file:
    #                 existing_q_table = pickle.load(file)
    #             merged_q_table = self.merge_q_tables(existing_q_table, current_q_table)
    #         else:
    #             merged_q_table = current_q_table
    #         with open(filename, 'wb') as file:
    #             pickle.dump(merged_q_table, file)
    #         self.logger.info(f"Saved Q-table to {filename}.")


    def merge_q_tables(self, existing_table, new_table):
        for state, actions in new_table.items():
            if state in existing_table:
                for action, weight in actions.items():
                    if action in existing_table[state]:
                        existing_table[state][action] = max(existing_table[state][action], weight)
                    else:
                        existing_table[state][action] = weight
            else:
                existing_table[state] = actions
        return existing_table            



    def initiate_game(self):
        self.startGame()
        self.pellets_numEaten = 0

    # method that gets the last knwon q table -> use this as a starting point
    def get_last_q_table(self):
        # see what is the latest qtable there is
        qtable_folder = 'Qtables'
        last_dictionary = sorted(list(os.listdir(qtable_folder)))[-1]
        dir = f'{qtable_folder}/{last_dictionary}'
        dir = f'{qtable_folder}/Qtable_2024-04-28 18:08:18.458029.pkl'

        # open the qtable and use that as the starting qtable
        with open(dir, 'rb') as f: loaded_q_table = pickle.load(f)
        return loaded_q_table
    
    # method that generates a small random number as a weight for an action
    def initialize_q_values(self, actions):
        return {action: np.random.uniform(0.01, 0.1) for action in actions}    
    

    # method that slowly decreases the chanes of generating random actions (not the action but a number - the action part comes later)
    def get_exploration_rate(self, episode, total_episodes, start=1.0, end=0.01, decay_rate=0.01):
        if episode > total_episodes/2: return start * (end / start) ** (episode / total_episodes) # slowly less randoms
        else: return end + (start - end) * np.exp(-decay_rate * episode / total_episodes)    # many many randoms

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
        # pac_pell_dists = [tuple(np.abs(np.array(i)-np.array(self.pacman_position))) for i in self.pellets_positions+self.powerpellets_positions]
        pac_pell_dists = [tuple(np.abs(np.array(i)-np.array(self.pacman_position))) for i in self.pellets_positions]
        pac_pell_dists = {i:np.sqrt(i[0]**2+i[1]**2) for i in pac_pell_dists}
        pac_pell_dists = {k:v for k,v in sorted(pac_pell_dists.items(),key=lambda i:i[1])}

        top_n_pellets = 4
        if len(pac_pell_dists) > top_n_pellets: closest_pellets = list(pac_pell_dists.keys())[:top_n_pellets]
        else: closest_pellets = pac_pell_dists
        
        if self.comments:
            print('\n\tself.pacman --> ',self.pacman_position)     
            print('\tself.ghosts --> ',self.ghosts_positions)
            print('\tself.closest_pellets --> ',closest_pellets)        

        # add the pacman and ghosts positions
        state.append(tuple(self.pacman_position))
        state.append(tuple(self.ghosts_positions))
        
        # get the ghosts statess
        for i in self.ghosts.ghosts:
            # if i.mode.current == 0: mode = 'scatter'
            # elif i.mode.current == 1: mode = 'chase'
            # elif i.mode.current == 2: mode = 'freight'
            # elif i.mode.current == 3: mode = 'spawn'
            # state.append(mode)
            state.append(i.mode.current)

        # Closest pellets' positions, also ensure a fixed length
        for i in range(top_n_pellets):
            try: state.append(closest_pellets[i])
            except: state.append((np.nan,np.nan))  # Use a placeholder if there are fewer pellets
        
        # Add all power pellents. Add all missing spaces if required
        for i in range(top_n_pellets):
            try: state.append(self.powerpellets_positions[i])
            except: state.append((np.nan,np.nan))  # Use a placeholder if there are fewer pellets
                
        return tuple(state)


    # Assume we have a function that can get all possible actions for a given state
    def getLegalActions(self,pacman_position:tuple):
        # Returns all legal actions for a given state
        if pacman_position in self.nodes_neighbors: 
            self.directions_valid = [k for k in self.nodes_neighbors[pacman_position].keys() if k < 3 and k > -3]
            self.reached_node = True
        else: 
            # self.directions_valid = [self.pacman.direction,-1*self.pacman.direction,STOP]
            self.directions_valid = [self.pacman.direction,STOP]
            self.reached_node = False
        return self.directions_valid     
    

    def update_iteration(self,desired_action:int):
        # print('\n\t#### we are updating the game')
        if self.comments:
            print('\n\t#### we are updating the game')
        # dt = self.clock.tick(10)
        # dt = self.clock.tick(30) / 1000.0
        # dt = self.clock.tick(30) / 500.0
        # dt = self.clock.tick(30) / 400.0
        dt = self.clock.tick(30) / 400.0
        self.pellets.update(dt)

         # unpause the game in the case that it is paused and remove the text
        self.checkEvents()
        if self.pause.paused: 
            if not self.paused_check:
                self.pause.flip()                
                self.showEntities()
        
        if self.paused_check: 
            # print('we are in qlearning')
            # print('self.pause.paused --> ',self.pause.paused)            
            with open(f'Qtables/Qtable_{os.getpid()}.pkl', 'wb') as f:
                pickle.dump(self.Q_table, f)   

        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

            if self.reset_level_check and self.comments:
                print('self.reset_level_check from qlearning.py --> ',self.reset_level_check)
            
        # check if a pellet has been eaten
        if self.pellets_numEaten != self.pellets.numEaten:
            self.pellets_numEaten += 1
            self.atePellet = True
        else: self.atePellet = False        

        
        if not self.pause.paused:
            self.pacman.update(dt,action=desired_action)        
        
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
        # choose to use an old q table or not
        if self.use_last_saved_q_table: Q_table = self.get_last_q_table()
        else: Q_table = {}
        
        actions_random_ratio_list,iterations_per_episode_list = [],[]
        self.actions_in_positions = {}
        actions = []

        state = self.getStateRepresentation()
        self.nodes_neighbors = {}
        for k,v in self.nodes.nodesLUT.items():
            self.nodes_neighbors[v.position.asTuple()] = {i:j for i,j in v.neighbors.items() if j is not None}
        # print('nodes_neighbors --> ',self.nodes_neighbors)   


        # Training loop
        for episode in range(self.total_episodes):            
            self.restart_game_check = False # reset this every time a new episode starts
            # print(f'\n---------- {episode+1}/{self.total_episodes} -------------- ')            
            state = self.getStateRepresentation()
            og_state = tuple(state)
            if state not in Q_table:                    
                Q_table[state] = self.initialize_q_values(self.getLegalActions(pacman_position=self.pacman.position.asTuple()))

            if not self.pacman.alive:
                if self.comments:
                    print('\tPacMan is dead. Restarting the game!')
                    print('\tself.lives --> ',self.lives)    
                
            # while pacman is alive, train the model. If dies, reset level and proceed. Do this until all episodes are done
            iteration_number = 0
            while self.pacman.alive:
                # if iteration_number > 1: [][0]                      
                self.textgroup.alltext[READYTXT].visible = False     # hide the text
                if self.restart_game_check: break                    # if we are starting a new game, break the loop and go to the next episode          

                # print(f'\n\nNew iteration {iteration_number} in episode {episode} \t{self.lives}/5')
                # print('\tstate --> ',state)
                # print('\tdirection --> ',self.pacman.direction)

                # if self.pacman.position.asTuple() in self.nodes_neighbors:
                    # print('.-.-.-.-.-.-.-.-..-.-.-.-.-.-.-.-..-.-.-.-.-.-.-.-. pacmans neighbors --> ',self.nodes_neighbors[self.pacman.position.asTuple()])   


                # find the action to tkae
                rand = np.random.rand()                    
                # this function gives us a rate that is to be used to determine if we do a random action or not (given that the prob is higher or below it)
                # the closer it is to the end episode, the more likely it will not do a random action                
                exploration_rate = self.get_exploration_rate(episode, self.total_episodes, start=1.0, end=0.01, decay_rate=0.01)
                
                # get the valid actions at the present time                    
                current_valid_actions = self.getLegalActions(pacman_position=self.pacman.position.asTuple())
                # pick an action to do
                if rand < exploration_rate or state not in Q_table:
                    if self.pacman.position.asTuple() not in self.actions_in_positions:
                        random.shuffle(current_valid_actions)
                        self.actions_in_positions[self.pacman.position.asTuple()] = {p:0 for p in current_valid_actions}                     
                    
                    freq_actions = {k:v for k,v in sorted(self.actions_in_positions[self.pacman.position.asTuple()].items(),key=lambda i:i[1],reverse=False) if k != STOP}
                    # print('\tvalid_actions --> ',current_valid_actions)
                    # print('\tself.actions_in_positions --> ',self.actions_in_positions[self.pacman.position.asTuple()])
                    # print('\tfreq_actions --> ',freq_actions)
                    
                    least_freq_action = list(freq_actions.keys())[0]

                    if random.random() < 0.7: # chance that pacman does not turn around (undo action done)
                        # print('\tæææ we are forcing a random action --> ',least_freq_action)
                        # desired_action = least_freq_action
                        if len(actions) > 3:
                            avoid_undo = [i for i in freq_actions.keys() if -1*i != actions[-3]]
                            if len(avoid_undo) != 0: desired_action = random.choice(avoid_undo)
                            else: desired_action = random.choice(list(freq_actions.keys()))
                        else: desired_action = random.choice(list(freq_actions.keys()))
                    else:                        
                        desired_action = random.choice([c for c in current_valid_actions if c != STOP])
                        # print('\twe are not forcing a random action --> ',desired_action)
                    actions_random_ratio_list.append('random')           

                    try:
                        self.actions_in_positions[self.pacman.position.asTuple()][desired_action] += 1
                    except Exception as e:
                        # print(f'Could not find action {desired_action} in {self.actions_in_positions[self.pacman.position.asTuple()]}')
                        print(e)
                        self.actions_in_positions[self.pacman.position.asTuple()][desired_action] = 1

                else:
                    desired_action = max(Q_table[state], key=Q_table[state].get)
                    actions_random_ratio_list.append('best')
                
                # IN THE CASE WHERE HAVE REACHED THE DESIRED TARGET, THERE WILL HAVE ALREADY BEEN SET A NEW TARGET USING AN ACTION SET BEFORE REACHING THE TARGET. SET TARGET TO DEPEND ON THE NEWLY DESIRED ACTION
                if self.reached_node:
                    self.pacman.direction = desired_action
                    self.pacman.target = self.pacman.getNewTarget(self.pacman.direction)

                # Take action, observe new state and reward
                self.Q_table = Q_table
                self.update_iteration(desired_action=desired_action)
                
                # set the action taken as the direction the model us heading todawrds to
                # if desired_action == self.pacman.direction: print('\t\tWe are heading our desired direction!!!  :))')
                # else: print('\t\tWe are not heading on our desired action. Waiting to arrive to target node  :/)')
                self.action = self.pacman.direction
                actions.append(self.action)    
                if len(actions) > 10: actions = actions[-10:] # ensure we keep the list of actions short
                # print('\taction to take ----> ',self.action)            
                # print('\tactions ----> ',actions)                                         

                # check if we are restarting level. If not, then proceed as expected. Otherwise skip this iteration
                if not self.reset_level_check:
                    newState = self.getStateRepresentation()
                    reward = self.getReward() # i have no idea how to do this one
                    # update the Q table witht the reward to the action

                    # Q-learning update
                    if newState not in Q_table:
                        new_valid_actions = self.getLegalActions(pacman_position=self.pacman.position.asTuple())
                        Q_table[newState] = self.initialize_q_values(new_valid_actions)
                    # print('\n\tnewState --> ',newState)
                    # print('\tQ_table[newState] --> ',Q_table[newState])
                        # Q_table[newState] = self.initialize_q_values(valid_actions)

                    # update the Q table using the QLearning formula
                    # try:
                    # try:
                        # print('\n\tQ_table state --> ',Q_table[og_state],self.action)                    
                        # print('\tQ_table new state --> ',Q_table[newState],self.action)
                        # print('\tQ_table state --> ',Q_table[og_state][self.action])
                    # except:
                        # print('\t\tQ_table error --> ',Q_table[og_state],'\tvalid_actions --> ',current_valid_actions,'\taction --> ',self.action)
                    try:                        
                        new_reward = (1 - self.alpha) * Q_table[og_state][self.action] + \
                                                self.alpha * (reward + self.gamma * max(Q_table[newState].values()))
                        new_reward_check = True
                    except:
                        try:
                            if og_state not in Q_table:
                                Q_table[og_state] = self.initialize_q_values(current_valid_actions)
                            new_reward = (1 - self.alpha) * Q_table[og_state][self.action] + \
                                                    self.alpha * (reward + self.gamma * max(Q_table[newState].values()))
                            new_reward_check = True
                        except: new_reward_check = False
                    
                    if new_reward_check:
                        # print(f'\t\tAction: {self.action}\tReward: {new_reward}')
                        Q_table[state][self.action] = new_reward                

                        # print out what needs to be printed if it is required
                        if self.comments:
                            print('\n\tstate --> ',state)
                            print('\t\tvalid_actions --> ',self.getLegalActions(pacman_position=self.pacman.position.asTuple()))
                            print('\t\t#####self.action --> ',self.action)
                            print('self.reset_level_check after the update function --> ',self.reset_level_check)                
                            print('\t\tnewState --> ',newState)                
                            print('\t\treward --> ',reward)
                            print('\t\tQ_table')
                            counter = 0
                            for k,v in Q_table.items(): 
                                if counter > len(Q_table)-5:                        
                                    print(f'\t\t\t{k} --> ',v)
                                counter += 1
                            
                        # Transition to the new state
                        state = newState
                        iteration_number += 1

                if self.reset_level_check:
                    iterations_per_episode_list.append(iteration_number)
                    # reset the info for the new life round
                    self.reset_level_check = False
                    iteration_number = 0        

                            
            # if allowed, plot tperformance and save dictionaryevery 10% episode
            if self.plot_performance:
                if episode%round(self.total_episodes*0.01) == 0 and episode != 0:
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
                    plt.savefig(f"Plots/{os.getpid()}.png", format='png')
            else:
                if episode%round(self.total_episodes*0.01) == 0 and episode != 0:
                    # Second plot for viewing the durection of the lives (the more the better)
                    plt.ioff()  # Turn the interactive mode off
                    plt.figure()  # Create a new figure
                    plt.plot(iterations_per_episode_list)
                    plt.title('Iterations per episode')

                    # Save the plot with a unique filename based on the process ID
                    filename = f"Plots/{os.getpid()}.png"
                    plt.savefig(filename, format='png')
                    print(f"Plot saved as {filename}")

                    plt.close()  # Close the plot to free up system resources

            if self.save_qtable:
                if episode%round(self.total_episodes*0.01) == 0 and episode != 0:
                    # save the Q table as a pickle file                    
                    with open(f'Qtables/Qtable_{os.getpid()}.pkl', 'wb') as f:
                        pickle.dump(Q_table, f)   
                                    
        # Run the game with the learned policy
        self.Q_table = Q_table
        
        if self.save_qtable:
            with open(f'Qtables/Qtable_{os.getpid()}.pkl', 'wb') as f:
                pickle.dump(self.Q_table, f)               




if __name__ == "__main__":
    ql = QLearning()
    # ql.getStateRepresentation()
    ql.initiate_game()
    ql.train()
    current_time = str(datetime.datetime.now())
    print(f'Ended running at time: {current_time}')
    print('\n\n',ql.Q_table.values())
                

    