# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:22:14 2019

@author: laurieshaw
"""

import Metrica as metrica
import Tracking_Visuals as vis

# IO - up date to your own filepaths
DATADIR = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Metrica/metrica_2018_full"
PLOTDIR = '/Users/laurieshaw/Documents/Football/Data/TrackingData/Metrica/plots/'
gameid = 217

# team names
hometeam = 'Orlando City SC'
awayteam = 'Toronto FC'

# read data 
frames, match, team1_players, team0_players = metrica.read_metrica_match_data(DATADIR,gameid)

# find corners in dataset
corners = match.events[ (match.events.Type=='SET PIECE') & (match.events.Subtype=='CORNER KICK')]

# plot player positions at first corner
corner_frame_number = match.frame_id_idx_map[corners.iloc[0]['Start Frame']] # need to use the id_idx_map to find the right frame in the list of frames
frame = frames[corner_frame_number] # get the relevant frame (or 'instant')

# make a plot of the frame (origin is at the center of the pitch)
vis.plot_frame(frame,match,units=1.0,include_player_velocities=False,include_ball_velocities=False)

# plot trajectories of attacking players in 3 second window around the corner, starting one second before the corner is taken

# first get the frames from 1 second before the corner to 2 seconds after
corner_frames = frames[corner_frame_number-25:corner_frame_number+2*25] # factor of 25 because there are 25 frames/second

penalty_area_edge = match.fPitchXSizeMeters/2.-16.38 # 16.38 is 18 yards in meters
penalty_area_width = 40.04 # 40.04 is 44 yards in meters

fig,ax = vis.plot_pitch(match)
vis.plot_frame(corner_frames[0],match,units=1.0,include_player_velocities=False,include_ball_velocities=False,figax=(fig,ax))

for cf in corner_frames:
    if corners.iloc[0]['Team']==hometeam:
        players = cf.team1_players # home team players
        pcolor = 'r.' # for plots
        direction_of_play = match.period_parity[frame.period]*-1 # 1 means home team is playing right->left, -1 means left-right (but switch sign for this example)
    else:
        players = cf.team0_players # away team players
        pcolor = 'b.'
        direction_of_play = match.period_parity[frame.period]
    # iterate over attacking players and plot their positions 
    for k in players.keys():
        # is player in penalty area, if so plot their position.
        if players[k].pos_x*direction_of_play > penalty_area_edge and abs(players[k].pos_y) < penalty_area_width/2.:
            ax.plot(players[k].pos_x*100,players[k].pos_y*100,pcolor,markersize=1)
            
# save a movie of the corner
vis.save_match_clip(corner_frames,match,PLOTDIR,fname='corner1',include_player_velocities=False,units=1.)
    