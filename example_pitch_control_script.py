# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:15:55 2019

@author: laurieshaw
"""

import Tracab as tracab
import Tracking_Visuals as vis
import numpy as np
import Pitch_Control as pc

fpath='/Path/To/Directory/of/Tracking/Data/' # path to directory of Tracab data
match_id = 984455 # example
   
print match_id
fname = str(match_id)
# read frames, match meta data, and data for individual players
frames_tb, match_tb, team1_players, team0_players = tracab.read_tracab_match_data('DSL',fpath,fname,verbose=False)

params = pc.default_model_params()
params['lambda_def'] = 3.99 # reset this so it's the same as the defending team

frame = frames_tb[7995] # plot at randome frame. need to start from at least the 5th frame as velocities are uninitialised until then
ball_pos = np.array([frame.ball_pos_x,frame.ball_pos_y])/100.

# if you want to actually take into acount the time taken for the ball to get to any given area of the pitch, you'll need to uncomment the lines below. 
# NB: calc_trajectory_grid, which generates a library of trajectories, takes a while to run (but you only need to generate it once)
# dgrid just calculates the shortest flight time for the ball to travel a certain distance.
'''
rgrid = pc.calc_trajectory_grid()
dgrid = pc.calc_shortest_flighttimes_array(np.arange(0,100,0.5),rgrid,dr=0.5,ball_vmax=40.)
'''

if frame.ball_team=='H':
    attacking_players = frame.team1_players
    defending_players = frame.team0_players
elif frame.ball_team=='A':
    attacking_players = frame.team0_players
    defending_players = frame.team1_players
                
# calculate attacking and defending team pitch control maps. PPCFtau shows how long a player from the attacking team would take to get to each point on the pitch. xgrid and ygrid indicate the pixel positions
# setting ball_pos = None assumes that the ball would instantaneously arrive at any given point on the pitch, otherwise set it = to ball_pos calculated above
PPCFa,PPCFd,PPCFtau,xgrid,ygrid = pc.generate_pitch_control_map(attacking_players, defending_players, frame, match_tb, params, ball_pos=None, dgrid=None, dT=0.01,tol=0.99 ) 

fig,ax = vis.plot_frame(frame,match_tb,include_player_velocities=True,include_ball_velocities=False)
ax.imshow(np.flipud(PPCFa), extent=(np.amin(xgrid)*100, np.amax(xgrid)*100, np.amin(ygrid)*100, np.amax(ygrid)*100),interpolation='hanning',vmin=0.2,vmax=1,cmap='spring',alpha=0.75)
