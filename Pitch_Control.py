# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:16:40 2019

@author: laurieshaw
"""

import numpy as np
from scipy.optimize import minimize

""" BALL TRAJECTORY ROUTINES """

def calc_ball_trajectory(init_speed,init_pitch_deg, alpha, dt = 0.01, maxiter=10000):
    # calculates a ball trajectory (distance, height and velocity as a function of time) given initial ball speed (m/s) and pitch (angle from ground, degrees)
    # accounts for air resistance 'alpha', so need so tbe solved numerically, using timestemps dt 
    g = 9.8 # m s^-2 # graviational constant
    r = np.zeros( shape=(maxiter,2), dtype=float) # 0: distance 1: height
    v = np.zeros( shape=(maxiter,2), dtype=float) # 0: horizontal speed 1: vertical speed
    init_pitch_radians = init_pitch_deg*np.pi/180.
    v[0,:] = init_speed*np.array( [np.cos(init_pitch_radians), np.sin(init_pitch_radians) ] )
    t = dt*np.arange(0,maxiter)
    counter = 1
    rdot = init_speed
    # now solve iteratively
    while r[counter-1,1]>=0: # stop when ball hits ground
        r[counter,:] = r[counter-1,:] + v[counter-1,:] * dt
        v[counter,1] = v[counter-1,1] - (g + alpha*rdot*v[counter-1,1])*dt
        v[counter,0] = v[counter-1,0] - alpha*rdot*v[counter-1,0]*dt
        rdot = np.linalg.norm(v[counter,:])
        counter += 1
    r = r[:counter-1,:]
    v = v[:counter-1,:]
    t = t[:counter-1]
    return r,v,t
       
def calc_ball_alpha(method='basic'):
    # holds air resistance parameter
    if method=='basic':
        mass = 0.42# kg
        rho = 1.22 # kg/m^3
        CD = 0.17 #
        A = 0.038 # m^2
        alpha = 1/2./mass*rho*CD*A
    return alpha
        
def calc_trajectory_grid(vmax = 40.0, dt = 0.01, maxiter=10000):
    # precomputes a dictionary of trajectories (a look up table) as a function of pass distance so that we don't repeatedly solve the ball trajectory during the pitch control calculation
    alpha = calc_ball_alpha()
    rmin = 0.0 # min distance
    rmax = 150 # max distance - approximately the length of diagonal from one corner of pitch to the opposite
    dr = 0.5 # steps in distance to calculate
    dtheta = 0.1 # steps in angle to calculate (degrees)
    theta_max = 80. # maximum pitch
    dv = 0.1
    v_init = np.arange(dv,vmax+dv,dv)
    theta_init = np.arange(dtheta,theta_max,dtheta)
    Nv = len(v_init)
    Nth = len(theta_init)
    # build trajectory grid
    rgrid = {}
    for r in np.arange(rmin,rmax+rmin,dr):
        rgrid[r] = []
    for i in range( Nv ):
        for j in range( Nth ):
            r,v,t = calc_ball_trajectory( v_init[i], theta_init[j], alpha, dt=dt )
            if r[-1,0]>rmax:
                print( "trajectory distance out of bounds" )
                assert False
            rkey = np.floor( r[-1,0]/dr ) * dr
            rgrid[rkey].append( (t[-1],v_init[i], theta_init[j]) )
    for r in rgrid.keys():
        rgrid[r] = sorted(rgrid[r],key = lambda x: x[0])
    return rgrid

def calc_shortest_flighttimes_array(distances,rgrid,dr=0.5,ball_vmax=40.):
    # calculate the shortest flight time for each distance overy a range of pass lengths
    dgrid = {}
    for distance in distances:
        rkey = np.floor( distance /dr ) * dr
        trajectories = [ t for t in rgrid[rkey] if t[1]<=ball_vmax ]
        if len(trajectories)>0:
            dgrid[rkey] = trajectories[0][0]
    return dgrid

""" PLAYER TRAJECTORY CALCULATIONS """ 


# USE y component of velocity in rotated frame to increase distance (and leave at that)
def piecewise_intercept_time(r_init,r_final,v_init,amax=7.,vmax=5., tmin=0.0):
    # This is a very approximated method for calculating the time taken for a player to get from A->B (r_init to r_final), with initial velocity v_init
    # r_init, r_final and v_init are all 2D vectors (x,y)
    th = lambda x: np.pi if x<0 else 0.
    r12 = r_final-r_init # vector from inital position to target position
    r12m = np.linalg.norm( r12 ) # distance from initial position to target position
    vtot = np.linalg.norm( v_init ) # initial speed
    theta = np.arctan( r12[1]/r12[0] ) + th(r12[0])
    # rotate co-ordiante system so that x-axis is aligned with r12
    R = np.array([ [np.cos(theta),np.sin(theta)], [-np.sin(theta),np.cos(theta)] ] )
    # now rotate velocity vector so that r12m is along the +x axis
    r12 = np.array([r12m,0.]) # vector from inital position to target position in rotated co-ordinate system
    v12 = np.matmul(R,v_init)
    t = 0
    if vtot>vmax: # slow down
        t += (vtot-vmax)/amax
        r12 -= v12*t - 0.5*amax*t*t*v12/vtot # new vector to ball
        r12m = np.linalg.norm( r12 )
        v12 = v12*vmax/vtot
        vtot = np.linalg.norm( v12 )
    if v12[0] < 0.0: # initially going in the wrong direction: decelerate to halt
        t += np.abs( v12[0] )/amax
        r12 -= v12*t - 0.5*amax*t*t*v12/vtot # new vector to ball
        r12m = np.linalg.norm( r12 )
        v12[0] = 0.0
        vtot = np.linalg.norm( v12 )
    tt = 1.9*np.abs(v12[1])/amax  # stopping time in transverse direction, 1.9 is a 'tuning parameter' to calibrate the approximation
    if np.abs(vmax**2-v12[0]**2)/2./amax < r12m:
        # distance travelled before top speed is reached
        tp = (vmax-v12[0])/amax + (r12m-max(vmax**2-v12[0]**2,0.)/2./amax)/vmax 
    else:
        tp = -v12[0]/amax + np.sqrt( v12[0]**2 + 2*amax*r12m )/amax
    t += np.sqrt(tt**2+tp**2) # approximately acceleration in parallel direction plus a penalty for transverse motion
    return t
  
def piecewise_intercept_time2(r_init,r_final,v_init,amax=7.,vmax=5., tmin_ball=0.0, reset_vmax=True):
    # This is a very approximated method for calculating the time taken for a player to get from A->B (r_init to r_final), with initial velocity v_init
    # r_init, r_final and v_init are all 2D vectors (x,y)
    th = lambda x: np.pi if x<0 else 0.
    r12 = r_final-r_init # vector from inital position to target position
    r12m = np.linalg.norm( r12 ) # distance from initial position to target position
    vtot = np.linalg.norm( v_init ) # initial speed
    if reset_vmax:
        vmax = max(vmax,vtot)
    if r12m==0:
        theta = 0.0
    else:
        theta = np.arctan( r12[1]/r12[0] ) + th(r12[0])
    # rotate co-ordiante system so that x-axis is aligned with r12
    R = np.array([ [np.cos(theta),np.sin(theta)], [-np.sin(theta),np.cos(theta)] ] )
    # now rotate velocity vector so that r12m is along the +x axis
    r12 = np.array([r12m,0.]) # vector from inital position to target position in rotated co-ordinate system
    v12 = np.matmul(R,v_init)
    t = 0
    if v12[0] < 0.0: # initially going in the wrong direction: decelerate to halt
        t += np.abs( v12[0] )/amax
        r12 -= v12*t - 0.5*amax*t*t*v12/vtot # new vector to ball
        r12m = np.linalg.norm( r12 )
        v12[0] = 0.0
        vtot = np.linalg.norm( v12 )
        
    if vtot>vmax: # slow down
        t += (vtot-vmax)/amax
        r12 -= v12*t - 0.5*amax*t*t*v12/vtot # new vector to ball
        r12m = np.linalg.norm( r12 )
        v12 = v12*vmax/vtot
        vtot = np.linalg.norm( v12 )
        
    tt = 1.9*np.abs(v12[1])/amax  # stopping time in transverse direction, 1.9 is a 'tuning parameter' to calibrate the approximation

    dstop = v12[0]*v12[0]/2./amax
    if dstop>r12[0]:
        tp = -v12[0]/amax + np.sqrt( v12[0]**2 + 2*amax*r12m )/amax
        if tp-tmin_ball<0:
            tp =  v12[0]/amax - np.sqrt( v12[0]**2 - 2*amax*r12m )/amax
        if tp-tmin_ball<0:
            tp =  v12[0]/amax + np.sqrt( v12[0]**2 - 2*amax*r12m )/amax
        
    elif np.abs(vmax**2-v12[0]**2)/2./amax < r12m:
        # distance travelled before top speed is reached
        tp = (vmax-v12[0])/amax + (r12m-max(vmax**2-v12[0]**2,0.)/2./amax)/vmax 
    else:
        tp = -v12[0]/amax + np.sqrt( v12[0]**2 + 2*amax*r12m )/amax
        
    t += np.sqrt(tt**2+tp**2) # approximately acceleration in parallel direction plus a penalty for transverse motion
    return t


def initialise_player_positions(attacking_players,defending_players,units=100.):
    for p in attacking_players.keys():
        attacking_players[p].r_init = np.array( [ attacking_players[p].pos_x, attacking_players[p].pos_y] ) / units # convert to m if units=100
        attacking_players[p].v_init = np.array( [ attacking_players[p].vx, attacking_players[p].vy ] )
    for p in defending_players.keys():
        defending_players[p].r_init = np.array( [ defending_players[p].pos_x, defending_players[p].pos_y] ) / units # convert to m if units=100
        defending_players[p].v_init = np.array( [ defending_players[p].vx, defending_players[p].vy ] )
    return attacking_players,defending_players

def get_optimal_arrival_time(tau_min,distance,dgrid, dr=0.5):
    # determines whether ball or player can arrive at a given point on the pitch first
    dgridmax = np.max(dgrid.keys()) # maximum distance in look-up table for pass trajectories
    rkey = np.floor( distance /dr ) * dr
    if rkey>dgridmax:
        return dgrid[dgridmax]
    elif dgrid[rkey]<tau_min:
        # ball can arrive faster than player, so use player arrival time
        return tau_min
    else:
        # player arrives before ball, so use fastest ball arrival time
        return dgrid[rkey]

def probability_intercept_ball(T, tau, s=0.54):
    # probability of a player controlling the ball, as described in Spearman 2018
    f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/s * (T-tau) ) )
    return f

""" Generate pitch control map """

def default_model_params():
    # key parameters for the model, as described in Spearman 2018
    params = {}
    params['max_player_accel'] = 7. # m/s/s
    params['max_player_speed'] = 5. # m/s
    params['control_sigma'] = 0.54
    params['lambda_att'] = 3.99
    params['lambda_def'] = 3.99*1.72
    params['max_ball_speed'] = 30. # m/s
    return params

def generate_pitch_control_map(attacking_players, defending_players, frame, match, params, ball_pos, dgrid, dT=0.005, tol=0.99, units=100.):
    # break the pitch down into a grid
    xgrid = np.linspace(-match.fPitchXSizeMeters/2.,match.fPitchXSizeMeters/2.,50)
    ygrid = np.linspace( -match.fPitchYSizeMeters/2.,match.fPitchYSizeMeters/2., 50*68/105.2 )
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFtau = np.zeros( shape = (len(ygrid), len(xgrid)) )
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    attacking_players,defending_players = initialise_player_positions(attacking_players,defending_players,units=units)
    # calculate pitch pitch control model at each location on the pitch
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            a,d,tau = calculate_pitch_control(target_position, attacking_players, defending_players, frame, ball_pos, dgrid, params, dT=dT, tol=tol, units=units)
            PPCFa[i,j] = a[-1]
            PPCFd[i,j] = d[-1]
            PPCFtau[i,j] = tau
    return PPCFa,PPCFd,PPCFtau,xgrid,ygrid

def calculate_pitch_control(target_position, attacking_players, defending_players, frame, ball_pos, dgrid, params, dT=0.005, tol=0.99,init_postions=False, units=100.):
    # core routine of the model. Calculates pitch control for attacking and defending teams at any given location on the pitch
    # get parameters
    amax = params['max_player_accel']
    vmax = params['max_player_speed']
    control_sigma = params['control_sigma']
    lambda_att = params['lambda_att']
    lambda_def = params['lambda_def']
    tau_min_att = 10000 # set to an arbitrarity high number
    tau_min_def = 10000 # set to an arbitrarity high number
    dTmax = 8.0 # in seconds
    akeys = attacking_players.keys()
    dkeys = defending_players.keys()
    
    if ball_pos is None: # assume that ball is already at location
        tmin_ball = 0.0 # or something non-zero but small?
    else:
        pass_dist = np.linalg.norm( target_position - ball_pos )
        tmin_ball = pass_dist/params['max_ball_speed']
        
    if init_postions: # player positions/velocities are not already initialised
        attacking_players,defending_players = initialise_player_positions(attacking_players,defending_players,units=units)
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    for p in akeys:
        tmin = piecewise_intercept_time2(attacking_players[p].r_init,target_position,attacking_players[p].v_init,amax=amax,vmax=vmax,tmin_ball=tmin_ball)
        attacking_players[p].exp_tau = tmin
        tau_min_att = min((tau_min_att, tmin))
        attacking_players[p].PPCF = 0.0
    for p in dkeys:
        tmin = piecewise_intercept_time2(defending_players[p].r_init,target_position,defending_players[p].v_init,amax=amax,vmax=vmax,tmin_ball=tmin_ball)
        defending_players[p].exp_tau = tmin
        tau_min_def = min((tau_min_def, tmin))
        defending_players[p].PPCF = 0.0
        
    # calculate optimal ball trajectory to pitch location
    if ball_pos is None: # assume that ball is already at location
        ball_arrival_time = dT
    else:
        ball_arrival_time = get_optimal_arrival_time(tau_min_att,pass_dist, dgrid, dr=0.5)
        
    # if defending team can arrive significantly before attacking team, no need to solve pitch control model
    if tau_min_att-max(ball_arrival_time,tau_min_def) >= 3*np.log(10)/lambda_def:
        PPCFatt = np.zeros( 1 )
        PPCFdef = np.ones( 1 )
        return PPCFatt, PPCFdef, tau_min_att
    # if attacking team can arrive significantly before defending team, no need to solve pitch control model
    elif tau_min_def-max(ball_arrival_time,tau_min_att) >= 3*np.log(10)/lambda_att:
        PPCFatt = np.ones( 1 )
        PPCFdef = np.zeros( 1 )
        return PPCFatt, PPCFdef, tau_min_att
    else: # solve pitch control model
        dT_array = np.arange(ball_arrival_time-dT,ball_arrival_time+dTmax,dT) 
        NT = len(dT_array)
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # calcaulte pitch control
        ptot = 0.0
        i = 1
        #for i in np.arange( 1, len( dT_array ) ):
        while i<NT and ptot<tol: # when total probability > 0.995, model solved.
            T = dT_array[i]
            for p in akeys:
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*probability_intercept_ball( T, attacking_players[p].exp_tau, s=control_sigma) * lambda_att
                assert dPPCFdT>=0
                attacking_players[p].PPCF = dPPCFdT*dT + attacking_players[p].PPCF
                PPCFatt[i] += attacking_players[p].PPCF
            for p in dkeys:
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*probability_intercept_ball( T, defending_players[p].exp_tau, s=control_sigma) * lambda_def
                assert dPPCFdT>=0
                defending_players[p].PPCF = dPPCFdT*dT + defending_players[p].PPCF
                PPCFdef[i] += defending_players[p].PPCF
            ptot = PPCFdef[i]+PPCFatt[i]
            i += 1
        return PPCFatt[:i], PPCFdef[:i], tau_min_att

def calc_xG_grid():
    strat_to_m = 0.267*0.91
    params_f=[-0.3664,-0.0335/strat_to_m, 0.0160]
    posts = 7.32 # 7.32 is width of goal line in m
    nx = 50
    ny = int( 50*68/105.2 )
    dx = 105.2/float(nx)
    dy = 68/float(ny)
    x = 105.2/2.-np.abs( np.linspace(-105.2/2.+dx/2.,105.2/2.-dx/2,nx) )
    y = np.linspace( -68/2.+dy/2.,68/2.-dy/2., ny )
    xgrid, ygrid = np.meshgrid(x, y)
    dgrid = np.sqrt(xgrid*xgrid+ygrid*ygrid) # distrance grid
 
    # y distance to posts (nearest,furthest)
    y_to_posts = (np.abs(np.abs(ygrid)-posts/2.),np.abs(ygrid)+posts/2.)
    # some angles to posts
    alpha = np.arctan(y_to_posts[0]/xgrid)
    phi = np.arctan(y_to_posts[1]/xgrid)
    # opening angle to goal
    angle_opening = np.zeros_like(xgrid)
    outpost = np.abs(ygrid)>posts/2.
    angle_opening[outpost] = (phi[outpost] - alpha[outpost])*180/np.pi
    angle_opening[~outpost] = (phi[~outpost] + alpha[~outpost])*180/np.pi
    
    X = params_f[0]*np.ones_like(dgrid) + params_f[1]*dgrid + params_f[2]*angle_opening
    expG = 1./(1.+np.exp( -X ))
    return expG,angle_opening,dgrid


def PPCF_match_clip_pass(side,frames,match,fpath,subsample=2,fname='PC_test'):
    # saves a movie of frames with filename 'fname' in directory 'fpath'
    xgrid = np.linspace(-105.2/2.,105.2/2.,50)
    ygrid = np.linspace( -68/2.,68/2., int( 50*68/105.2 ) )
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='test clip!')
    writer = FFMpegWriter(fps=int(match.iFrameRateFps/float(subsample)), metadata=metadata)
    fig,ax = vis.plot_pitch(match)
    points_on = False
    team1 = np.zeros((14,4))
    team0 = np.zeros((14,4))
    fname = fpath + fname + '.mp4'
    
    params = default_model_params()
    
    frames = frames[::subsample]
    with writer.saving(fig, fname, 100):
        for frame in frames:
            print( frame.timestamp )
            if points_on:
                pts1.remove()
                pts2.remove()
                pts3.remove()
                pts4.remove()
                pts6.remove()
                pts7.remove()
                pts8.remove()
            for i,j in enumerate( frame.team1_jersey_nums_in_frame ):
                team1[i,0] = frame.team1_players[j].pos_x
                team1[i,1] = frame.team1_players[j].pos_y
                team1[i,2] = frame.team1_players[j].vx/2.
                team1[i,3] = frame.team1_players[j].vy/2.
            team1 = team1[:i+1,:]
            for i,j in enumerate( frame.team0_jersey_nums_in_frame ):
                team0[i,0] = frame.team0_players[j].pos_x
                team0[i,1] = frame.team0_players[j].pos_y
                team0[i,2] = frame.team0_players[j].vx/2.
                team0[i,3] = frame.team0_players[j].vy/2.
            team0 = team0[:i+1,:]
            pts1, = ax.plot( team1[:,0], team1[:,1], 'ro',linewidth=0,markeredgewidth=0,markersize=10)
            pts2, = ax.plot( team0[:,0], team0[:,1], 'bo',linewidth=0,markeredgewidth=0,markersize=10)
            pts6 = ax.quiver( team1[:,0], team1[:,1], team1[:,2], team1[:,3],color='r',width=0.002,headlength=5,headwidth=3,scale=80)
            pts7 = ax.quiver( team0[:,0], team0[:,1], team0[:,2], team0[:,3],color='b',width=0.002,headlength=5,headwidth=3,scale=80)
            
            # add pitch control
            if side=='H':
                attacking_players = frame.team1_players
                defending_players = frame.team0_players
            elif side=='A':
                attacking_players = frame.team0_players
                defending_players = frame.team1_players
            PPCFa,PPCFd,PPCFtau,xgrid,ygrid = generate_pitch_control_map(attacking_players, defending_players, frame, params, ball_pos=None, dgrid=None , dT=0.05, tol=0.95 )
            pts8 = ax.imshow(np.flipud(PPCFa), extent=(np.amin(xgrid)*100, np.amax(xgrid)*100, np.amin(ygrid)*100, np.amax(ygrid)*100),interpolation='None',vmin=0.2,vmax=1,cmap='spring',alpha=0.3)
 
            
            if frame.ball and frame.ball_status=='Alive':
                pts3, = ax.plot( frame.ball_pos_x, frame.ball_pos_y, marker='o',markerfacecolor='k',markeredgewidth=0,markersize=6*max(1,np.sqrt(frame.ball_pos_z/150.)))
            else:
                pts3, = ax.plot( 0, 0, marker='x',markerfacecolor='k',markeredgewidth=0,markersize=6) 
            pts4 = ax.text(-150,match.fPitchYSizeMeters*50+40,frame.min+':'+ frame.sec.zfill(2), fontsize=14 )
            if not points_on:
                points_on = True
            writer.grab_frame()


    

    
    
    
    
    
    
    
    
    
    
    