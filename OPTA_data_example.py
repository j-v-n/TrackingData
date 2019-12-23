# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:49 2019

@author: laurieshaw
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:33:43 2019

@author: laurieshaw
"""

import OPTA as opta
import OPTA_visuals as ovis


fpath = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/SuperLiga/All/"
#fpath='/n/home03/lshaw/Tracking/Tracab/SuperLiga/' # path to directory of Tracab data


match_id = 984458
fname = str(match_id)

match_OPTA = opta.read_OPTA_f7(fpath,fname) 
match_OPTA = opta.read_OPTA_f24(fpath,fname,match_OPTA)

ovis.plot_all_shots(match_OPTA,plotly=False)
ovis.make_expG_timeline(match_OPTA)

# print shots
shots = [e for e in match_OPTA.events if e.is_shot]
homeshots = [e for e in match_OPTA.hometeam.events if e.is_shot]
awayshots = [e for e in match_OPTA.awayteam.events if e.is_shot]

for s in shots:
    print s