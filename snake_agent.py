import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #  N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        print("IN helper_func")
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        if snake_head_x <= 40:
            adjoining_wall_x = 1
        elif snake_head_x >= 480:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0
        
        if snake_head_y <= 40:
            adjoining_wall_y = 1
        elif snake_head_y >= 480:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        if snake_head_x <= 0 or snake_head_y <= 0 or snake_head_x >= 520 or snake_head_y >= 520:
            adjoining_wall_x = 0
            adjoining_wall_y = 0

        if snake_head_x > food_x:
            food_dir_x = 1
        elif snake_head_x < food_x:
            food_dir_x = 2
        else:
            food_dir_x = 0

        if snake_head_y > food_y:
            food_dir_y = 1
        elif snake_head_y < food_y:
            food_dir_y = 2
        else:
            food_dir_y = 0

        if(snake_head_x, snake_head_y - 40) in snake_body:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if(snake_head_x, snake_head_y + 40) in snake_body:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0
        
        if(snake_head_x - 40, snake_head_y) in snake_body:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if(snake_head_x + 40, snake_head_y) in snake_body:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        curr_state = (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

        return curr_state


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the reinforcement learning agent.
    #   it decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        print("IN AGENT_ACTION")
        prime = self.helper_func(state)
        action = self.a

        reward = self.compute_reward(points, dead)

        if self._train:
            if self.s != None:
                next_q = np.max(self.Q[prime])
                alpha = self.LPC / (self.LPC + self.N[self.s + (action,)])
                curr_q = self.Q[self.s + (action,)]
                self.Q[self.s + (action,)] += alpha * (reward + self.gamma * next_q - curr_q)
            
            max_val = float('-inf')
            best_action = 0
            for a in range(3, -1, -1):
                if self.N[prime + (a, )] < self.Ne:
                    curr_f = 1
                else:
                    curr_f = self.Q[prime + (a,)]
                if curr_f > max_val:
                    max_val = curr_f
                    best_action = a
            if not dead:
                self.N[prime + (best_action, )] += 1
                self.points = points
            self.s = prime
            self.a = best_action

        else:
            best_action = np.argmax(self.Q[prime])
        if dead:
            self.reset()
            
        return best_action
