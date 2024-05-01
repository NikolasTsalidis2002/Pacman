import os
import pickle
import numpy as np
from run import GameController
import matplotlib.pyplot as plt
import datetime
from constants import *
import random

from collections import Counter


class QLearning(GameController):
    def __init__(self, test_q_table=False, save_qtable=False, use_last_saved_q_table=True, plot_performance=True, use_flee_seek_behaviour=False ,comments=False):
        super().__init__(initial_pause_check=False)

        # Q-learning parameters
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        # self.epsilon = 0.9  # exploration rate
        self.total_episodes = 3

        # do not touch these!
        self.paused_check = False        
        self.reset_level_check = False
        self.restart_game_check = False

        self.use_flee_seek_behaviour = use_flee_seek_behaviour

        # these are just the arguments. Change the arguments, not the variables
        self.save_qtable = save_qtable
        self.use_last_saved_q_table = use_last_saved_q_table
        self.plot_performance = plot_performance
        self.comments = comments
        self.test_q_table = test_q_table


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
        # directory = f'{qtable_folder}/{last_dictionary}'
        directory = f'{qtable_folder}/{last_dictionary}'

        # open the qtable and use that as the starting qtable
        with open(directory, 'rb') as f: loaded_q_table = pickle.load(f)
        return loaded_q_table
    
    # method that generates a small random number as a weight for an action
    def initialize_q_values(self, actions):
        return {action: np.random.uniform(0.01, 0.1) for action in actions}    
    

    # method that slowly decreases the chanes of generating random actions (not the action but a number - the action part comes later)
    def get_exploration_rate(self, episode, total_episodes, start=1.0, end=0.01, decay_rate=0.01):
        if episode > total_episodes/2: return start * (end / start) ** ((total_episodes/2+episode) / total_episodes) # slowly less randoms
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
        self.directions_valid = self.pacman.directions
        return self.directions_valid     
    

    def update_iteration(self, desired_action:int): # the desired action comes from the training. Only activated when not using seek_flee_behavour
        # print('\n\t#### we are updating the game')
        self.ateGhost = False
        # dt = self.clock.tick(10)
        # dt = self.clock.tick(30) / 1000.0
        # dt = self.clock.tick(30) / 400.0     
        dt = self.clock.tick(30) / 300.0     
        self.dt = dt

        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

            if self.reset_level_check and self.comments:
                print('self.reset_level_check from qlearning.py --> ',self.reset_level_check)

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, desired_action=desired_action)
        else:
            self.pacman.update(dt, desired_action=desired_action)   

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

        # check if a pellet has been eaten
        if self.pellets_numEaten != self.pellets.numEaten:
            self.pellets_numEaten += 1
            self.atePellet = True
        else: self.atePellet = False        

           

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


    # function that gets the actions for learning. It can be random or be the best
    def getAction(self,Q_table:dict,state:str):
        # find the action to tkae
        rand = np.random.rand()                    
        # this function gives us a rate that is to be used to determine if we do a random action or not (given that the prob is higher or below it)
        # we can use it for training and improving the qtable or testing the qtable
        if not self.test_q_table: exploration_rate = self.get_exploration_rate(self.episode, self.total_episodes, start=1.0, end=0.01, decay_rate=0.01)
        else: exploration_rate = self.get_exploration_rate(episode=self.total_episodes,total_episodes=self.total_episodes, start=1.0, end=0.01, decay_rate=0.01)

        
        # get the valid actions at the present time                            
        current_valid_actions = list(self.current_legal_actions.keys())
        print('current_valid_actions --> ',current_valid_actions)

        # pick an action. Random is random number says so ot the state is not in the qtable
        print(rand,exploration_rate)
        if rand < exploration_rate or state not in Q_table:
            # this actions is going to be random
            if self.pacman.position.asTuple() not in self.actions_in_positions:
                random.shuffle(current_valid_actions) # shuffle the actions so they keys are not in order
                self.actions_in_positions[self.pacman.position.asTuple()] = {p:0 for p in current_valid_actions}                     
            
            # we have an issue, and that is that Pacman has a very high chance of going back on its steps. To avoid this, we reduce the chances of allowing this to happen
            # by setting a prov threshold (0.7) 
            freq_actions = {k:v for k,v in sorted(self.actions_in_positions[self.pacman.position.asTuple()].items(),key=lambda i:i[1],reverse=False) if k != STOP}

            if random.random() < 0.7: # chance that pacman does not turn around (undo action done)
                # print('\tæææ we are forcing a random action --> ',least_freq_action)
                # desired_action = least_freq_action
                if len(self.actions) > 3:
                    avoid_undo = [i for i in freq_actions.keys() if -1*i != self.actions[-3]]
                    if len(avoid_undo) != 0: desired_action = random.choice(avoid_undo)
                    else: desired_action = random.choice(list(freq_actions.keys()))
                else: desired_action = random.choice(list(freq_actions.keys()))
            else:                        
                desired_action = random.choice([c for c in current_valid_actions if c != STOP])
                # print('\twe are not forcing a random action --> ',desired_action)
            self.actions_random_ratio_list.append('random')    

            try:
                self.actions_in_positions[self.pacman.position.asTuple()][desired_action] += 1
            except Exception as e:
                # print(f'Could not find action {desired_action} in {self.actions_in_positions[self.pacman.position.asTuple()]}')
                print(e)
                self.actions_in_positions[self.pacman.position.asTuple()][desired_action] = 1

        else:
            # this action is going to to be one witht the most rewards in this state
            desired_action = max(Q_table[state], key=Q_table[state].get)
            self.actions_random_ratio_list.append('best')   
        print('øøøøøøøøøøøøøø This is the ratio of the best and random actions in our list',Counter(self.actions_random_ratio_list))
        return desired_action



    def train(self):
        # get the Qtable
        if self.use_last_saved_q_table: Q_table = self.get_last_q_table()
        else: Q_table = {}
        
        # iniate empty lists to keep track of things 
        self.actions_random_ratio_list,self.iterations_per_episode_list,self.actions = [],[],[]
        self.actions_in_positions = {}
        # get all the nodes' neighbors
        self.nodes_neighbors = {}
        for k,v in self.nodes.nodesLUT.items():
            self.nodes_neighbors[v.position.asTuple()] = {i:j for i,j in v.neighbors.items() if j is not None}            
        print('nodes_neighbors --> ',self.nodes_neighbors)

        # while True:
        for self.episode in range(self.total_episodes):   
            print(f'\n---------- {self.episode+1}/{self.total_episodes} -------------- ')
            iteration_number = 0 # keep track of the number of iterations that there are per episode
            self.restart_game_check = False 
            
            while not self.restart_game_check:
                # get the current state representation of the game and ensure it is one the Qtable
                state = self.getStateRepresentation()
                og_state = tuple(state)
                print('-----------\npacman position --> ',self.pacman.position)
                print('   state --> ',state)
                self.current_legal_actions = self.getLegalActions(pacman_position=self.pacman.position.asTuple())
                print('\tself.current_legal_actions --> ',self.current_legal_actions)
                if state not in Q_table:                    
                    Q_table[state] = self.initialize_q_values(self.current_legal_actions)
                
                # ensure that the legal actions are correct
                # print('Qtable last state --> ',Q_table[list(Q_table.keys())[-1]])
                if not self.pause.paused:
                    if len(Q_table) > 2:
                        if list(Q_table[list(Q_table.keys())[-2]].keys()) != list(Q_table[list(Q_table.keys())[-1]].keys()):
                            print('Change in actions --> ',Q_table[list(Q_table.keys())[-1]])                        
  
                if not self.use_flee_seek_behaviour: desired_action = self.getAction(Q_table=Q_table,state=state)
                else: desired_action = None       

                print(f'===================== self.use_flee_seek_behaviour: {self.use_flee_seek_behaviour} \tdesired_action --> ',desired_action)   

                # update iteration
                self.update_iteration(desired_action=desired_action)

                # once updated, keep track of the actions done by looking at the current direction
                self.action = self.pacman.direction
                print('\taction --> ',self.action)
                self.actions.append(self.action)    
                if len(self.actions) > 10: self.actions = self.actions[-10:] # ensure we keep the list of actions short

                # if not self.reset_level_check:s
                reward = self.getReward() 

                # update the pacmans position for the new state representiation
                if not self.pause.paused: self.pacman.position += self.pacman.directions[self.pacman.direction]*self.pacman.speed*self.dt
                newState = self.getStateRepresentation()                
                # update the Q table witht the reward to the action

                # Q-learning update
                if newState not in Q_table:
                    new_valid_actions = self.getLegalActions(pacman_position=self.pacman.position.asTuple())
                    Q_table[newState] = self.initialize_q_values(new_valid_actions)
                
                print('pacman position --> ',self.pacman.position)
                print('og_state --> ',og_state)
                new_reward = (1 - self.alpha) * Q_table[og_state][self.action] + \
                                        self.alpha * (reward + self.gamma * max(Q_table[newState].values()))
                
                print(f'\t\tAction: {self.action}\tReward: {new_reward}')
                Q_table[og_state][self.action] = new_reward                

                # Transition to the new state
                state = newState
                iteration_number += 1


            # if allowed, plot tperformance and save dictionaryevery 10% episode
            if self.total_episodes > 100: per = 0.01
            if self.total_episodes < 100: per = 0.1
            if self.total_episodes < 10: per = 1                        
            if self.plot_performance:            
                # if self.episode%round(self.total_episodes*per) == 0 and self.episode != 0:
                # First plot for viewing the action randomness ditribution
                plt.figure(1)  # Create a new figure
                plt.hist(self.actions_random_ratio_list)
                plt.title(self.episode)
                plt.show(block=False)
                plt.pause(0.001)  # Pause for a brief moment to ensure the plot is rendered
                
                # Second plot for viewing the durection of the lives (the more the better)
                plt.figure(2)  # Create a new figure
                plt.plot(self.iterations_per_episode_list)
                plt.title('Iterations per episode')
                plt.show(block=False)
                plt.pause(0.001)  # Pause for a brief moment to ensure the plot is rendered
                plt.savefig(f"Plots/{os.getpid()}.png", format='png')
            else:
                if self.episode%round(self.total_episodes*per) == 0 and self.episode != 0:
                    # Second plot for viewing the durection of the lives (the more the better)
                    plt.ioff()  # Turn the interactive mode off
                    plt.figure()  # Create a new figure
                    plt.plot(self.iterations_per_episode_list)
                    plt.title('Iterations per episode')

                    # Save the plot with a unique filename based on the process ID
                    filename = f"Plots/{os.getpid()}.png"
                    plt.savefig(filename, format='png')
                    print(f"Plot saved as {filename}")

                    plt.close()  # Close the plot to free up system resources

            if self.save_qtable:
                with open(f'Qtables/Qtable_{os.getpid()}.pkl', 'wb') as f:
                    pickle.dump(Q_table, f)   
                                    
        # Run the game with the learned policy
        self.Q_table = Q_table
               

    def update_MergedQTables(self):    
        qtables_folder = 'Qtables'
        tables = [i for i in os.listdir(qtables_folder) if i != '.gitattributes']
                          
        # open the merged tables pkl file. If it does not exist, set as existing table an empty dict
        try: 
            with open(qtables_folder+'/MergedTables.pkl', 'rb') as file: existing_table = pickle.load(file)    
        except: existing_table = {}

        # go thorugh all the tables there are. Add them on to the existing table dict
        for t in tables:
            address = qtables_folder+'/'+t
            with open(address, 'rb') as file: table = pickle.load(file)    
            existing_table = ql.merge_q_tables(existing_table=existing_table,new_table=table)
            # remove the just merged file
            os.remove(address)
        # udpate the MergedTables.pkl file
        with open(qtables_folder+'/MergedTables.pkl', 'wb') as file: pickle.dump(existing_table, file)    




if __name__ == "__main__":
    test_q_table = True    
    if test_q_table: 
        print('TESTING Q TABLE')
        ql = QLearning(use_flee_seek_behaviour=False,test_q_table=True,plot_performance=True,
                                    save_qtable=False,use_last_saved_q_table=True)
        print('We are going to start the game now')
        ql.initiate_game()
        print('Game initiated')
        ql.train()
        
    else: 
        print('TRAINING AGENT')
        ql = QLearning(use_flee_seek_behaviour=True,test_q_table=False,plot_performance=False,
                       save_qtable=True,use_last_saved_q_table=False)
        print('We are going to start the game now')
        ql.initiate_game()
        print('Game initiated')
        ql.train()
        print('Training over. Updating MergedTables.pkl')
        ql.update_MergedQTables() # udapte the latest qtables
        
                

    