# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:06:54 2019

Module for reading Tracab data and constructing frame and player objects

@author: laurieshaw
"""

import datetime as dt
import numpy as np
import Tracking_Velocities as vel
import csv as csv
import pandas as pd
from os import listdir
  

''' BASIC IO ROUTINES '''

def get_metrica_events(DATA_DIR,event_file):
    # read metrica event data file into a dataframe
    event_df = pd.read_csv('{}/{}'.format(DATA_DIR, event_file))
    return event_df

def metrica_to_pandas(DATA_DIR,filename):
    # read metrica tracking data into a dataframe
    csvfile =  open('{}/{}'.format(DATA_DIR, filename), 'r')
    reader = csv.reader(csvfile)    
    team = next(reader)[3][:3].lower()
    # construct column names
    jerseys = [x for x in next(reader) if x != '']
    columns = next(reader)
    for i, j in enumerate(jerseys):
        columns[i*2+3] = "{}_x_{}".format(team, j)
        columns[i*2+4] = "{}_y_{}".format(team, j)
    columns[-1] = "ball_x"
    columns.append("ball_y")
    df = pd.read_csv('{}/{}'.format(DATA_DIR, filename), names=columns, skiprows=3)
    return df
    
def get_metrica_frames(DATA_DIR,tracking_files):
    dfs = [metrica_to_pandas(DATA_DIR,f) for f in tracking_files]
    # merge home & away team tracking data into single dataframe
    return dfs[0].drop(columns=['ball_x', 'ball_y']).merge(
            dfs[1], on=('Period', 'Frame', 'Time [s]'))

def get_filenames(DATA_DIR,game_id):
    # get tracking data and event data filenames for a given match
    all_files = listdir(DATA_DIR)
    tracking_files = [x for x in all_files if str(game_id) in x and 'Track' in x]
    assert len(tracking_files) == 2, "Wrong # of DFs"
    event_files = [x for x in all_files if str(game_id) in x and 'Events' in x]
    assert len(event_files)==1
    return tracking_files, event_files[0]

''' READ IN METRICA DATA AND CONVERT TO FRAME CLASS '''

def read_metrica_match_data(DATA_DIR, game_id, during_match_only=True, verbose=True):
    # get filenames
    if verbose:
        print "* Reading data"
    tracking_files, event_file = get_filenames(DATA_DIR,game_id)
    # get basic match data
    match = metrica_match(DATA_DIR,event_file)
    #  read in tracking data
    df = get_metrica_frames(DATA_DIR,tracking_files)
    if verbose:
        print "* Generating frames"
    frames = []
    for i,row in df.iterrows():
        frames.append( metrica_frame( row, match ) )  
    # timestamp frames
    if verbose:
        print "* Timestamping frames"
    frames, match = timestamp_frames(frames,match,during_match_only=during_match_only)
    # run some basic checks
    check_frames(frames)
    # identify which way each team is shooting
    set_parity(frames, match)
    # get player objects and calculate ball and player & team com velocity 
    if verbose:
        print "* Generating player structures"
    team1_players, team0_players = get_players(frames)
    if verbose:
        print "* Finding goalkeepers"
    team1_GK,team0_GK = get_goalkeeper_numbers(frames)
    match.team1_exclude = team1_GK
    match.team0_exclude = team0_GK
    if verbose:
        print "* Measuring velocities"
    vel.estimate_player_velocities(team1_players, team0_players, match, window=7, polyorder=1, maxspeed = 14, units=1.)
    vel.estimate_com_frames(frames,match,team1_GK,team0_GK)
    return frames, match, team1_players, team0_players

def set_parity(frames, match):
    # determines the direction in which the home team are shooting
    # 1: right->left, -1: left->right
    match.period_parity = {}
    team1_x = np.mean([frames[0].team1_players[pid].pos_x for pid in frames[0].team1_jersey_nums_in_frame])
    if team1_x > 0:
        match.period_parity[1] = 1 # home team is shooting from right->left in 1st half
        match.period_parity[2] = -1
    else:
        match.period_parity[1] = -1 # home team is shooting from left->right in 1st half
        match.period_parity[2] = 1
    return match

def check_frames(frames):
    # some very basic checks on the frames
    frameids = [frame.frameid for frame in frames]
    missing = set(frameids).difference(set(range(min(frameids),max(frameids)+1)))
    nduplicates = len(frameids)-len(np.unique(frameids))
    if len(missing)>0:
        print "Check Fail: Missing frames"
    if nduplicates>0:
        print "Check Fail: Duplicate frames found"

def get_goalkeeper_numbers(frames,verbose=True):
    # find goalkeeper jersey numbers based on average playere positions
    ids = frames[0].team1_jersey_nums_in_frame
    x = []
    for i in ids:
        x.append( frames[0].team1_players[i].pos_x )
    team1_exclude = [ ids[np.argmax( np.abs( x ) )] ]
    # check to see if there is a goalkeeper sub
    if team1_exclude[0] not in frames[-1].team1_jersey_nums_in_frame:
        team1_exclude = team1_exclude + find_GK_substitution(frames,1,team1_exclude[0])
    ids = frames[0].team0_jersey_nums_in_frame
    x = []
    for i in ids:
        x.append( frames[0].team0_players[i].pos_x )
    team0_exclude = [ ids[np.argmax( np.abs( x ) )] ]
    if team0_exclude[0] not in frames[-1].team0_jersey_nums_in_frame:
        team0_exclude = team0_exclude + find_GK_substitution(frames,0,team0_exclude[0])
    if verbose:
        print "home goalkeeper(s): ", team1_exclude
        print "away goalkeeper(s): ", team0_exclude
    return team1_exclude, team0_exclude

def find_GK_substitution(frames,team,first_gk_id):
    # check to see if there was a GK substitution
    gk_id = first_gk_id
    new_gk = []
    plast = []
    for frame in frames[::25]: # look every second
        if team==1:
            pnums = frame.team1_jersey_nums_in_frame
        else:
            pnums = frame.team0_jersey_nums_in_frame
        if gk_id not in pnums:
            sub = set(pnums) - set(plast)
            print sub
            if len(sub)!=1:
                print "No or more than one substitute"
                assert False
            else:
                new_gk.append( int(list(sub)[0]) )
                gk_id = new_gk[-1]
        plast = pnums
    if len(new_gk) != 1:
        print "goalkeeper sub problem", new_gk
    return new_gk

def get_players(frames):
    # first get all players that appear in at least one frame
    # get all jerseys in frames
    team1_jerseys = set([]) # home team 
    team0_jerseys = set([]) # away team
    team1_players = {}
    team0_players = {}
    for frame in frames:
        team1_jerseys.update( frame.team1_jersey_nums_in_frame )
        team0_jerseys.update( frame.team0_jersey_nums_in_frame )
    for j1 in team1_jerseys:
        team1_players[j1] = metrica_player(j1,1)
    for j0 in team0_jerseys:
        team0_players[j0] = metrica_player(j0,0)
    for frame in frames:
        for j in team1_jerseys:
            if j in frame.team1_jersey_nums_in_frame:
                team1_players[j].add_frame(frame.team1_players[j],frame.frameid,frame.timestamp)
            else: # player is not in this frame
                team1_players[j].add_null_frame(frame.frameid,frame.timestamp)
        for j in team0_jerseys:
            if j in frame.team0_jersey_nums_in_frame:
                team0_players[j].add_frame(frame.team0_players[j],frame.frameid,frame.timestamp)
            else: # player is not in this frame
                team0_players[j].add_null_frame(frame.frameid,frame.timestamp)
    return team1_players, team0_players

def timestamp_frames(frames,match,during_match_only=True):
    # remove frames before the beginning of the match or during half time
    # Frames must be sorted into ascending frameid first
    for i,frame in enumerate(frames):
        if frame.period==1:
            if frame.frameid<match.period_attributes[1]['iStartFrame']:
                frame.timestamp = -1
                frame.period = -1 # before match 
            else:
                frame.timestamp = frame.rawtimestamp-match.period_attributes[1]['StartTime']/60.
                frame.min = str( int(frame.timestamp) )
                frame.sec = "%1.2f" % ( round((frame.timestamp-int(frame.timestamp))*60.,3) )
        elif frame.period==2:
            if frame.frameid<match.period_attributes[2]['iStartFrame']:
                frame.timestamp = -1
                frame.period = -2 # during half time
            else:
                frame.timestamp = frame.rawtimestamp-match.period_attributes[2]['StartTime']/60.
                frame.min = str( int(frame.timestamp) )
                frame.sec = "%1.2f" % ( round((frame.timestamp-int(frame.timestamp))*60.,3) )  
    if during_match_only:
        frames = [f for f in frames if f.period>0]
        # quick lookup for finding the right frame from the event data
        for i,f in enumerate(frames):
            match.frame_id_idx_map[f.frameid] = i
    return frames, match


# match class: holds basic information about the match, including the event data
class metrica_match(object):
    def __init__(self,DATADIR,eventfile):
        self.provider = 'Metrica'
        self.DATADIR = DATADIR
        self.eventfile = eventfile
        datestring = eventfile.split('_')[-2]
        self.date = dt.datetime.strptime(datestring, '%Y%m%d')
        self.hometeam = str.lower(eventfile.split('_')[3])
        self.awayteam = str.lower(eventfile.split('_')[4])
        print "%s vs %s on %s" % (self.hometeam,self.awayteam,datestring)
        self.fPitchXSizeMeters = 105. # pitch length in m (assumed as not in data)
        self.fPitchYSizeMeters = 68. # pitch width in m
        self.iFrameRateFps = 25. # frames per second
        self.frame_id_idx_map = {}
        self.add_event_data()
        
    def add_event_data(self):
        events = pd.read_csv('{}/{}'.format(self.DATADIR, self.eventfile))
        self.events = events
        self.period_attributes = {}
        # now find time and frame when each half starts
        first_half_start_idx = events['Subtype'].eq('KICK OFF').idxmax()
        first_half_start_frame = events['Start Frame'].loc[first_half_start_idx]
        first_half_start_time = events['Start Time [s]'].loc[first_half_start_idx]
        second_half_start_idx = events[ (events['Period']==2) & (events['Subtype']=='KICK OFF') ].index[0]
        second_half_start_frame = events['Start Frame'].loc[second_half_start_idx]
        second_half_start_time = events['Start Time [s]'].loc[second_half_start_idx]       
        self.period_attributes[1] = {'iStartFrame':first_half_start_frame,'StartTime':first_half_start_time}
        self.period_attributes[2] = {'iStartFrame':second_half_start_frame,'StartTime':second_half_start_time}
        
# frame class: holds tracking data at a given time instant (frame)
class metrica_frame(object):
    def __init__(self,rowdata,match):
        self.provider = 'Metrica'
        self.frameid = int( rowdata['Frame'] )
        self.period = int( rowdata['Period'] )
        self.rawtimestamp = float( rowdata['Time [s]'] )/60. # convert to minutes
        self.team1_players = {}
        self.team0_players = {}
        self.team1_jersey_nums_in_frame = []
        self.team0_jersey_nums_in_frame = []
        # find jersey numbers in row
        homesquad_jersey = np.unique( [int(col.split('_')[-1]) for col in rowdata.keys() if col.startswith(match.hometeam)] )
        awaysquad_jersey = np.unique( [int(col.split('_')[-1]) for col in rowdata.keys() if col.startswith(match.awayteam)] )
        
        # iterate through home players and add to frame
        for j in homesquad_jersey:
            x = rowdata[match.hometeam+'_x_'+str(j)]
            y = rowdata[match.hometeam+'_y_'+str(j)]
            if not ( np.isnan( x ) or np.isnan( y ) ):
                self.add_frame_target( 1, j, x, y,match)
        # iterate through away players and add to frame
        for j in awaysquad_jersey:
            x = rowdata[match.awayteam+'_x_'+str(j)]
            y = rowdata[match.awayteam+'_y_'+str(j)]
            if not ( np.isnan( x ) or np.isnan( y ) ):
                self.add_frame_target( 0, j, x, y,match) 
        # add ball
        if np.isnan( rowdata['ball_x'] ) or np.isnan( rowdata['ball_y'] ):
            self.ball = False
        else:
            self.add_frame_ball(rowdata['ball_x'],rowdata['ball_y'],match)
        
        
    def add_frame_target(self,team_id,jersey_num,pos_x,pos_y,match):
        # add a player to the frame
        team_id = team_id 
        jersey_num =jersey_num
        # move to native co-ordinate system (in cms, with origin at the center circle)
        pos_x = ( pos_x-0.5 ) * match.fPitchXSizeMeters
        pos_y = ( pos_y-0.5 ) * match.fPitchYSizeMeters
        speed = np.nan
        if team_id==1:
            self.team1_players[jersey_num] = metrica_target(team_id,jersey_num,pos_x,pos_y,speed) 
            self.team1_jersey_nums_in_frame.append( jersey_num )
        elif team_id==0:
            self.team0_players[jersey_num] = metrica_target(team_id,jersey_num,pos_x,pos_y,speed) 
            self.team0_jersey_nums_in_frame.append( jersey_num )
         
    
    def add_frame_ball(self,ball_x,ball_y,match):
        self.ball = True
        self.ball_pos_x = ( ball_x-0.5 ) * match.fPitchXSizeMeters
        self.ball_pos_y = ( ball_y-0.5 ) * match.fPitchYSizeMeters
        self.ball_pos_z = 0.0 # metrica doesn't have z-axis for ball, so put on ground in all frames
        
        
    def __repr__(self):
        nplayers = len(self.team1_jersey_nums_in_frame)+len(self.team1_jersey_nums_in_frame)
        s = 'Frame id: %d, nplayers: %d, nballs: %d' % (self.frameid, nplayers, self.ball*1)
        return s
        
class metrica_target(object):
    # defines position of an individual target 'player' in a frame
    def __init__(self,team,jersey_num,pos_x,pos_y,speed):
        self.team = team
        self.jersey_num = jersey_num
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.speed = speed
        
class metrica_player(object):
    # contains trajectory of a single player over the entire match
    def __init__(self,jersey_num, teamID):
        self.jersey_num = jersey_num
        self.teamID = teamID
        self.frame_targets = []
        self.frame_timestamps = []
        self.frameids = []
        
    def add_frame(self,target,frameid,timestamp):
        self.frame_targets.append( target )
        self.frame_timestamps.append( timestamp )
        self.frameids.append( frameid )
        
    def add_null_frame(self,frameid,timestamp):
        # player is not on the pitch (specifically, is not in a frame)
        self.frame_timestamps.append( timestamp )
        self.frameids.append( frameid )
        self.frame_targets.append( metrica_target(self.teamID, self.jersey_num, 0.0, 0.0, None ) )
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
