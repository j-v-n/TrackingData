# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:20:04 2016

@author: laurieshaw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import unicodedata
from matplotlib.patches import Ellipse

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii 

def movingave_adaptive(x,window):
    # applies a moving average with an adaptive window
    non_nan = ~np.isnan(x)
    xcut = x[non_nan]
    if window>len(xcut):
        print "window too large, or too many nans"
        return
    smoothed = np.convolve(xcut,np.ones(window)/float(window),'same')
    steps = int(window/2.)
    for i in range(0,steps):
        smoothed[i] =  np.mean(xcut[:2*i+1])
        smoothed[-(i+1)] =  np.mean(xcut[-(2*i+1):])
    # now put it all back together
    final = np.array( [np.nan]*len(non_nan), dtype=float )
    final[non_nan] = smoothed
    #plt.plot(x,'bd')
    #plt.plot(np.convolve(x,np.ones(window)/float(window),'same'),'b')
    #plt.plot(final,'go-')
    return final
    
def nancumsum_with_reset(x,reset=0.):
    # cumsum routine in which nans reset the sum to zero (or 'reset')
    csum = np.zeros(np.shape(x))
    last = 0.0
    for i in np.arange(0,len(x)):
        if np.isnan(x[i]):
            csum[i] = 0.0
        else:
            csum[i] = x[i] + last
        last = csum[i]
    return csum
            
def EPL_imscatter(x,y,teams,season='1718',ax=None,zoom=0.25,alpha=0.9):
    if not ax:
        fig,ax = plt.subplots()
    images = read_EPL_images(season=season)
    icons = [OffsetImage(images[t],zoom=zoom,alpha=alpha) for t in teams]
    imscatter(x, y, icons, ax=ax)

def read_EPL_images(season='1718'):
    fdir = "/Users/laurieshaw/Documents/Football/MiscPlots/Icons/" + season + "/"
    Teams = ['AFC Bournemouth','Arsenal','Brighton & Hove Albion','Burnley',
             'Chelsea','Crystal Palace','Everton','Huddersfield Town',
             'Leicester City','Liverpool','Manchester City','Manchester United',
             'Newcastle United','Southampton','Stoke City','Swansea City',
             'Tottenham Hotspur','Watford','West Bromwich Albion','West Ham United',
             'Hull City','Sunderland','Middlesbrough']
    images = {}
    for team in Teams:
        fpath = fdir + team.replace(" ","") + ".png"
        images[team] = plt.imread(fpath)
    return images
    
def imscatter(x, y, icons, ax=None):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, icon in zip(x, y, icons):
        ab = AnnotationBbox(icon, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
    
    
def plot_bivariate_normal(mean, cov, figax=None, nsigma=1, lt='ko', fc=None, alpha=0.3):
    fcolors = iter(['blue', 'red', 'green', 'black'])
    pcolors = iter(['bd', 'rd', 'gd', 'black'])
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nsigma * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=w, height=h, angle=theta)
    if fc is None:
        fc = next(fcolors)
    ell.set_facecolor(fc)
    ell.set_alpha(alpha)
    pt = ax.add_artist(ell)
    return fig, ax, pt
    
    
    
    
    
    
    