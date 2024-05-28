#import basic modules
import os
import numpy as np 
import random
from random import randint
from skimage.draw import line
import skimage
import cv2
import matplotlib.pyplot as plt
import sys
from shapely.geometry import LineString, Polygon, Point, LinearRing, MultiLineString
from shapely.plotting import plot_points, plot_line, plot_polygon

#import custom classes
from fem_solver import FEM



class env():
    def __init__(self, state_size): #state size is 64x64; grid_size is 32x32; board size is 20x20m

        self.fixed_stock = (np.random.uniform(2, 6, 20)).tolist() ###########################################################################################


        #initialise attributes
        self.state_size = state_size
        self.grid_size = 32
        self.board_size = 20 #m
        self.scale_factor = self.board_size/self.grid_size
        self.r_clr = 1 #state space rendering color

        self.end_position,self.stock_id, self.place_bool = None,None,None #self.end_position represents the cursor location
        self.action = None
        
        #initialise domain for boundary conditions
        self.support_domain = None #binary 2D matrix
        self.supports = []
        self.design_domain = None #float32 2D matrix
        
        #construct FEM object
        self.solver = FEM()

        # initialise parameters for grid space
        self.poslist = np.arange(self.grid_size*self.grid_size).reshape(self.grid_size,self.grid_size)

        #reset environment (+ initialise other attributes)
        self.reset()

    def reset(self):
        
        self.state = np.zeros((self.state_size[0],self.state_size[0]))
        self.noise = np.random.normal(size=(1,10))

        self.component_stock = np.copy(self.fixed_stock)#(np.random.uniform(1, 5, 20)).tolist() ###########################################################################################
        print(f'component_stock: {self.component_stock}')

        self.support_domain = np.zeros((self.state_size[0],self.state_size[0]))
        self.support_domain[:10,:] = 1
        self.support_domain[-10:,:] = 1
        self.supports = [[0,0]]
                
        # print('random_stock: ',self.component_stock)
        self.reward = 0
        self.placed_components = []
        self.actions_list = []
        self.fem_reward = 0
        self.penalty = 0
        self.max_disp = 0
        self.start_position = np.array([0,0]) 

        # self.state[tuple(self.current_position)] = 0.9 #make current position visible
        self.done = False
        self.found_supports_coords = []
        self.current_state = self.state
        self.next_state = None

    def place_component(self):#start_pos, next_loc, element_length, placement=True): #add third argument for placement_flag
        current_loc = self.start_position*self.scale_factor # these coordinates must be scaled#####################################################################################
        next_loc = self.end_position*self.scale_factor
        component_length = self.component_stock[self.stock_id]
        
        if (self.place_bool == True) and (component_length>0) and (next_loc.tolist() != current_loc.tolist()):
            
            direction = next_loc-current_loc

            unit_direction = direction/np.linalg.norm(direction)
            scaled_vector = unit_direction*component_length
            end_point = current_loc+scaled_vector # if placement flag is Flase, the end point is not computed and the function returns next_loc
            

            #build and store line object
            component = LineString((current_loc,end_point))
            
            self.placed_components.append(component)
            
            #update stock list (the selected element is 'zeroed')
            self.component_stock[self.stock_id] = 0

            #render line in canvas observation
            r_start = int(self.start_position[0]*2), int(self.start_position[1]*2)
            r_end = int((end_point[0]/self.scale_factor)*2), int((end_point[1]/self.scale_factor)*2)
            rr, cc = line(r_start[0], r_start[1], r_end[0], r_end[1])
            # print('coords = ' , np.int32(self.start_position*2), np.int32((end_point/self.scale_factor)*2))
            # print('component length = ', component_length)
            # print('r_end = ', r_end)
            rr = np.clip(rr,0,self.state_size[0]-1)
            cc = np.clip(cc,0,self.state_size[0]-1)
            self.state[rr, cc] = self.r_clr

            if  0 <= r_end[0] < self.state_size[0] and  0 <= r_end[1] < self.state_size[0]:
                if self.support_domain[r_end]==1: #works with any binary mask
                    self.supports.append(end_point.tolist())
                    #@@@@@@@@@@@@@@@@@ add reward @@@@@@@@@@@@@@@@@#
                    #self.reward = 0.1
                    # print(f'########################################## support added: {end_point}', f'    raster coord: {r_end}')
        elif (self.place_bool == True) and (component_length==0): #equivalent to place an element with length 0
            end_point = current_loc
        else:
            end_point = next_loc #jump to a location defined by the sampled action (which otherwise defines the component orientation line)
        
        return end_point


    def get_fem_result(self): #better renamed as compute rewards


        # print(f'supports: {self.supports}')
        self.solver.run(self.placed_components, supports = self.supports) #builds the model, solves intersections, assigns mechanical properties, runs the solver, returns nodal displacements
        self.max_disp = max(self.solver.nodal_displacements)
        #normalise fem output
        d_thresh = 200
        max_disp_norm = 1 if (self.max_disp >= d_thresh) or (self.max_disp == 0) else self.max_disp/d_thresh
        self.fem_reward = 1 - (max_disp_norm*1)
        print(f'********************************************************** fem reward is: {self.fem_reward}')        

        #compute substructures count penalty
        c_thresh = 10 #10 more than the analysed one
        substructure_count = self.solver.substructures_count-1
        connectivity_pen = 1 if (substructure_count >= c_thresh) else substructure_count/c_thresh
        self.penalty += connectivity_pen*0.5
        print(f'********************************************************** connectivity penalty is: {connectivity_pen}')

        #compute span_penalty (% of canvas)
        # Sum the values in each column
        column_sums = np.sum(self.state, axis=1)
        zero_columns_count = np.sum(column_sums == 0)
        span_penalty = zero_columns_count/self.state_size[0]
        print(f'********************************************************** span penalty is: {span_penalty}')
        self.penalty += span_penalty*0.5

        #compute final reward
        self.reward = self.fem_reward - self.penalty

    def step(self, action): #action domain is [32x32, stock_size=20?, 2] e.g. [500, 6, 1]

        self.reward = 0

        self.action = action
        self.actions_list.append(self.action)

        self.end_position = np.argwhere(self.poslist==self.action[0])[0] #cursor position
        self.stock_id = self.action[1]
        self.place_bool = self.action[2]

        self.current_state = np.copy(self.state) #somewhere I also have to update the stock array (each time an item is picked, the corresponding values for that member will be removed from the stock array/dictionary))

        #render line in the state space
        component_end_point = self.place_component() #if flag is true, creates a Line object and appends it to the placed components list. the end point is the component end. if flag is false, it simply returns the sampled action
        ######################################################### check wether the end_point(raster) is in support_domain. if it is contained, add the point(vector) to supports 
        self.start_position = component_end_point/self.scale_factor # non integer     
        
        #render cursor position in the observation space
        #self.state[tuple(self.end_position*2)] = 1
        
        if self.done:
            self.get_fem_result()
            print(f'structure analysed: max displacement is {self.max_disp}, computed reward (after penalty) is: {self.reward}')            
            # try:
            #     self.get_fem_result()
            #     print(f'structure analysed, max displacement is {self.reward}')
            # except:
            #     pass

        self.next_state = np.copy(self.state)


        return (self.current_state,self.next_state,self.component_stock,self.reward,self.done,self.noise)

######################################################################

    def plot(self,state,size,filename):
        Z = skimage.transform.resize(state,
                                    (size,size),
                                    mode='edge',
                                    anti_aliasing=False,
                                    anti_aliasing_sigma=None,
                                    preserve_range=True,
                                    order=0)
        cv2.imwrite(filename,np.rot90(Z)*255)
    
    
    
    def plot_state(self,input_array,plot_name):
        # cmap = plt.get_cmap("hsv")
        # norm = plt.Normalize(input_array.min(), input_array.max())


        save_dir = 'plots'+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.plot(input_array,512,os.path.join(save_dir,plot_name + '.png'))#,str(self.reward)+str(self.new_position)


#test the environment
if __name__ == "__main__":
    env = env([64,64])
    pf_list = [1,1,1,0,1,1,1,0,1,1]
    for j in range(10):
        env.reset()
        for i in range(10):
            pos =  random.randint(0,1023)
            pf =  pf_list[i]
            print(f'action: {(pos,i,pf)}')
            env.step([pos, i, pf])
        try:
            env.get_fem_result()
        except:
            print('***************************************** COULD NOT ANALYSE MODEL *****************************************')
    # env.reset()
    # env.step([500, 5, 1])
    # env.step([999, 19, 1])
        # env.get_fem_result()
        env.solver.visualise_model(view_deflected=False)
        env.solver.visualise_model(view_deflected=True)
        print(f'structure analysed, max displacement is {env.reward}')

    # print(received_data)
    # # print(env.persistent_obstacle_coords)

        state = env.next_state
        env.plot_state(state,f'plot no. {j}')
        # cv2.imshow('env',cv2.resize(np.rot90(state),(512,512),interpolation = cv2.INTER_NEAREST))
        # cv2.waitKey(0)
