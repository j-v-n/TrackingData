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
    teamnamefull = next(reader)[3].lower()
    teamnameshort = filename.split('_')[-2].lower()
    print(teamnamefull,teamnameshort)
    # use teamname from filename rat
    # construct column names
    jerseys = [x for x in next(reader) if x != '']
    columns = next(reader)
    for i, j in enumerate(jerseys):
        columns[i*2+3] = "{}_x_{}".format(teamnameshort, j)
        columns[i*2+4] = "{}_y_{}".format(teamnameshort, j)
    columns[-1] = "ball_x"
    columns.append("ball_y")
    df = pd.read_csv('{}/{}'.format(DATA_DIR, filename), names=columns, skiprows=3)
    return df
    
def get_metrica_frames(DATA_DIR,tracking_files):
    dfs = [metrica_to_pandas(DATA_DIR,f) for f in tracking_files]
    # merge home & away team tracking data into single dataframe
    return dfs[0].drop(columns=['ball_x', 'ball_y']).merge(
            dfs[1], on=('Period', 'Frame', 'Time [s]'))

def get_player_name_to_jersey_map(DATA_DIR,tracking_files):
    # read metrica tracking data into a dataframe
    playername_jerseynum_map = {}
    for tracking_file in tracking_files:
        csvfile =  open('{}/{}'.format(DATA_DIR, tracking_file), 'r')
        reader = csv.reader(csvfile)    
        teamnamefull = next(reader)[3]
        # construct player names to jersey map
        jerseys = [x for x in next(reader) if x != '']
        columns = next(reader)
        playernames = [x for x in columns[3:] if x not in ['','Ball']]
        playername_jerseynum_map[teamnamefull] = {n:c for n,c in zip(playernames,jerseys)}
    return playername_jerseynum_map

def get_filenames(DATA_DIR,game_id):
    # get tracking data and event data filenames for a given match
    all_files = listdir(DATA_DIR)
    tracking_files = [x for x in all_files if '_{0:03}_'.format(game_id) in x and 'Track' in x and 'merged' not in x]
    assert len(tracking_files) == 2, "Wrong # of DFs"
    event_files = [x for x in all_files if '_{0:03}_'.format(game_id) in x and 'Events' in x]
    assert len(event_files)==1
    return tracking_files, event_files[0]

''' READ IN METRICA DATA AND CONVERT TO FRAME CLASS '''

def read_metrica_match_data(DATA_DIR, game_id, during_match_only=True, verbose=True):
    # get filenames
    if verbose:
        print( "* Reading data" )
    tracking_files, event_file = get_filenames(DATA_DIR,game_id)
    # get basic match data
    match = metrica_match(DATA_DIR,event_file)
    # get player name to jersey number map for each team
    match.playername_jerseynum_map = get_player_name_to_jersey_map(DATA_DIR,tracking_files)
    #  read in tracking data
    df = get_metrica_frames(DATA_DIR,tracking_files)
    if verbose:
        print( "* Generating frames" )
    frames = []
    for i,row in df.iterrows():
        frames.append( metrica_frame( row, match ) )  
    # timestamp frames
    if verbose:
        print( "* Timestamping frames" )
    frames, match = timestamp_frames(frames,match,during_match_only=during_match_only)
    # run some basic checks
    check_frames(frames)
    # identify which way each team is shooting
    set_parity(frames, match)
    # get player objects and calculate ball and player & team com velocity 
    if verbose:
        print( "* Generating player structures" )
    team1_players, team0_players = get_players(frames)
    if verbose:
        print( "* Finding goalkeepers" )
    team1_GK,team0_GK = get_goalkeeper_numbers(frames)
    match.team1_exclude = team1_GK
    match.team0_exclude = team0_GK
    if verbose:
        print( "* Measuring velocities" )
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
        print( "Check Fail: Missing frames" )
    if nduplicates>0:
        print( "Check Fail: Duplicate frames found" )

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
        print( "home goalkeeper(s): ", team1_exclude )
        print( "away goalkeeper(s): ", team0_exclude )
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
            print( sub )
            if len(sub)!=1:
                print( "No or more than one substitute" )
                assert False
            else:
                new_gk.append( int(list(sub)[0]) )
                gk_id = new_gk[-1]
        plast = pnums
    if len(new_gk) != 1:
        print( "goalkeeper sub problem", new_gk )
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

def set_ball_possession_status(frames,match):
    events = match.events
    teams = set(events.Team)
    # find home and away team
    if match.hometeam=='tor':
        hometeam = 'Toronto FC'
        awayteam = list( teams.difference({'Toronto FC'}) )[0]
    elif match.awayteam=='tor':
        awayteam = 'Toronto FC'
        hometeam = list( teams.difference({'Toronto FC'}) )[0]
    else:
        assert False, "Toronto FC not in match"
    print("%s vs %s" % (hometeam,awayteam) )
    # event types
    pos_start_events = ['SET PIECE','RECOVERY']
    pos_end_events = ['BALL LOST','BALL OUT','FAULT RECEIVED','SHOT']
    other_events = ['CARD','CHALLENGE','PASS']
    # remove other_events from log
    events = events[~events['Type'].isin(other_events)].sort_values(by=['Start Frame', 'End Frame'])
    # seperate home and away events
    homeevents = events[events.Team==hometeam]
    awayevents = events[events.Team==awayteam]
    # estimate for number of frames
    nframes = len(frames)
    first_half_last_event_row = events['Period'].eq(2.0).idxmax()-1

    home_pos,home_bad = create_possession_log(homeevents,frames,match,first_half_last_event_row,'H')
    away_pos,away_bad = create_possession_log(awayevents,frames,match,first_half_last_event_row,'A')
    possessions = home_pos + away_pos
    bad_possessions = home_bad + away_bad
    
    # sort the possessions
    possessions = sorted(possessions,key=lambda x: x.pos_start_fnum)
    # deal with bad possessions
    for bp in bad_possessions:
        start_frame_number = bp[0]
        # find first possession that starts after this one
        next_possession = next( (x for x in possessions if x.pos_start_fnum>start_frame_number), None)
        if next_possession is None:
            print('No following possession after bad possession %s,%d' % (bp[1],start_frame_number))
        else:
            end_frame_number = next_possession.pos_start_fnum-1
            possessions.append( metrica_possession(frames[start_frame_number:end_frame_number],start_frame_number,bp[2],bp[1],None) )   
    # sort the possessions again
    possessions = sorted(possessions,key=lambda x: x.pos_start_fnum)
    # find frames in which the ball is dead (play has stopped)
    for i,p in enumerate(possessions[:-1]):
        prev_pos_end = p.pos_end_fnum+1 # frame after last frame of previous possession
        next_pos_start = possessions[i+1].pos_start_fnum
        if (next_pos_start>prev_pos_end) and (p.pos_end_type=='BALL OUT' or possessions[i+1].pos_start_type=='SET PIECE'):
            for frame in frames[prev_pos_end:next_pos_start]:  
                frame.ball_status = 'Dead'
                frame.ball_team = None
        else:
            for frame in frames[prev_pos_end:next_pos_start]:  
                frame.ball_status = 'Alive'
                frame.ball_team = frames[p.pos_end_fnum].ball_team if frame.ball_team is None else frame.ball_team# allocate possession to the preceding team in possession. 
    # additional information about possessions (type - attacking, defensive, neutral & start transitions, dead ball)
    for pos in possessions:
        pos.set_possession_type(frames,match)
    # print possession summary for the match
    ptype = np.zeros(nframes)    
    for i,frame in enumerate(frames):
        ptype[i] = 1.0 if frame.ball_team=='H' else -1.0 if frame.ball_team=='A' else 0.0 if frame.ball_team=='N' else np.nan
    ball_in_play = np.sum(~np.isnan(ptype)) / match.iFrameRateFps / 60. # in minutes
    home_team_pos = np.sum(ptype==1) / match.iFrameRateFps / 60.
    away_team_pos = np.sum(ptype==-1) / match.iFrameRateFps / 60. 
    pos_total = home_team_pos + away_team_pos
    print( "Possession Stats:")
    print( "Ball in play: %1.2f mins" % (ball_in_play) )
    print( "Home pos: %1.2f mins (%1.2f%%), Away pos: %1.2f mins (%1.2f%%)" % (home_team_pos,100*home_team_pos/pos_total,away_team_pos,100*away_team_pos/pos_total) )
    return possessions, frames

def create_possession_log(team_events,frames,match,first_half_last_event_row,team):
    pos_start_events = ['SET PIECE','RECOVERY']
    pos_end_events = ['BALL LOST','BALL OUT','FAULT RECEIVED','SHOT']
    possessions = []
    bad_possessions = []
    in_possession = False
    for i,row in team_events.iterrows():
        start_frame_number = match.frame_id_idx_map[ row['Start Frame'] ]
        if not in_possession:
            if row['Type'] in pos_start_events:
                in_possession = True
                pos_start_frame = start_frame_number
                start_event = row
            else:
                print( "Not a possession start event: %s (row: %d, frame: %d, home team)" % (row['Type'],i,start_frame_number) )
        else:
            if row['Type'] in pos_end_events or i==first_half_last_event_row:
                in_possession = False
                possessions.append( metrica_possession(frames[pos_start_frame:start_frame_number+1],pos_start_frame,team,start_event,row) )
            else:
                print( "Not a possession end event: %s (row: %d, frame: %d, home team)" % (row['Type'],i,start_frame_number) )
                # if new possession starts in a later frame to current one, flag older one as a bad possession
                # otherwise, revert to older one as start of the current possession
                if start_frame_number>pos_start_frame:
                    bad_possessions.append((pos_start_frame,start_event,team))
                    pos_start_frame = start_frame_number 
                    start_event = row
    return possessions,bad_possessions
    

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
        print( "%s vs %s on %s" % (self.hometeam,self.awayteam,datestring) )
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
        self.pos_phase = None
        self.def_phase = None
        
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
        self.ball_team = None # add using event data
        self.ball_status = None # add using event data
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
        pos_y = ( pos_y-0.5 ) * match.fPitchYSizeMeters*-1 # flip y-axis
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
        self.ball_pos_y = ( ball_y-0.5 ) * match.fPitchYSizeMeters*-1 # flip y-axis
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
        

class metrica_possession(object):
    # a single period of continuous posession for a team
    def __init__(self,frames,frame_start_num,team,start_event,end_event):
        self.team = team
        self.teamname = start_event['Team']
        self.start_event = start_event
        self.end_event = end_event
        self.period = frames[0].period
        self.pos_start_fid = frames[0].frameid
        self.pos_end_fid = frames[-1].frameid
        self.pos_start_time = frames[0].timestamp
        self.pos_end_time = frames[-1].timestamp
        self.pos_duration = 60*(self.pos_end_time - self.pos_start_time)
        self.pos_Nframes = len(frames)
        self.pos_start_fnum = frame_start_num
        self.pos_end_fnum = frame_start_num+self.pos_Nframes-1
        self.pos_start_type = start_event['Type']
        self.pos_end_type = 'Unknown' if end_event is None else end_event['Type']
        self.pos_end_subtype = 'Unknown' if end_event is None else end_event['Subtype']
        self.goal = 'GOAL' in str(self.pos_end_subtype).split('-')
        if self.pos_end_type=='SHOT' and 'OUT' in str(end_event['Subtype']):
            self.pos_end_type = 'BALL OUT'
        self.prev_pos_type = None
        self.pos_type = None
        # check if team changes during the possesion (suggesions a data error in ball status)
        self.bad_possesion = ( set( frames[0].team1_jersey_nums_in_frame ) != set( frames[-1].team1_jersey_nums_in_frame ) ) or ( set( frames[0].team0_jersey_nums_in_frame ) != set( frames[-1].team0_jersey_nums_in_frame ) )
        if self.bad_possesion:
            print("bad posession: period %d, %1.2f to %1.2f" % (self.period, self.pos_start_time,self.pos_end_time))
        assert frames[0].period==frames[-1].period, "Possessions must end at end of period %s,%d" % (self.team,self.pos_start_fid)
        # tag frames
        for f in frames:
            f.ball_status = 'Alive'
            if f.ball_team is None:
                f.ball_team = self.team
            elif f.ball_team==self.team:
                assert False, "Possesion frame overlap for same team %s,%d,%d" % (self.team,self.pos_start_fnum,self.pos_end_fnum)
            else:
                f.ball_team = 'N' # unclear who has possession
        
    def set_possession_type(self,frames,match):
        pos_frames = frames[self.pos_start_fnum:self.pos_end_fnum+1]
        if self.team=='H': # team shooting from right->left
            # team-based metric
            x = np.median( [f.team1_x for f in pos_frames] ) * match.period_parity[self.period]
        else: # team shooting from left->right
            # team-based metric
            x = np.median( [f.team0_x for f in pos_frames] ) * -1*match.period_parity[self.period]
        if x < 0:
            self.pos_type = 'A'
        elif x > match.fPitchXSizeMeters*100/4.:
            self.pos_type = 'D'
        else:
            self.pos_type = 'N' # neutral
        
    def __repr__(self):
        s = 'Team %s, Period %d, Type %s, start = %1.2f, end = %1.2f, length = %1.2f' % (self.team,self.period,self.pos_type,self.pos_start_time,self.pos_end_time,self.pos_duration)
        return s        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
