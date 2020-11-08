# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:01:01 2020

@author: Flat J
"""
import pandas as pd

birddata = pd.read_csv("bird_tracking.csv")

import matplotlib.pyplot as plt
import numpy as np

ix = birddata.bird_name == "Eric"
x, y = birddata.longitude[ix],birddata.latitude[ix]  #indexing using a boolean

plt.plot(x,y, "-")

bird_names = pd.unique(birddata.bird_name)

for name in bird_names:
    ix = birddata.bird_name == name
    x, y = birddata.longitude[ix],birddata.latitude[ix]  #indexing using a boolean
    plt.plot(x,y, "-", label=name)
plt.xlabel("Longitude")
plt.ylabel("Lattitude")
plt.legend(loc="lower right")
plt.savefig("3traj.pdf")
plt.clf()

ix = birddata.bird_name == "Eric"
speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0, 30, 20), density=True)
plt.xlabel("2D speed")
plt.savefig("speedhist.pdf")
plt.clf()

birddata.speed_2d.plot(kind='hist', range=[0,30])
plt.xlabel("2D speed")
plt.savefig("pd_speedhist.pdf")
plt.clf()

#using date-time
import datetime

#this code strips the date_time column of bird data and converts it to a time object
timestamps = []
for k in birddata.date_time:
    date_str = k
    time_obj = datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")
    timestamps.append(time_obj)

birddata["timestamp"] = pd.Series(timestamps, index=birddata.index) #initialises a new column to the  birddata DF

times = birddata.timestamp[birddata.bird_name == "Eric"]
elapsed_time = [time - times[0]  for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)

next_day = 1
inds = [] #indicies for a given  day
daily_mean_speed = []

for i, t in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i) #indicies for current day are collected
    else:
        daily_mean_speed.append(np.mean(birddata.speed_2d[inds])) #computes daily mean speed
        next_day += 1 # then updates the day
        inds = [] #then clears the inds for the new day
        
plt.plot(daily_mean_speed)
plt.xlabel("day")
plt.ylabel("mean speed")
plt.savefig("dms.pdf")
plt.clf()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.figure(figsize=(10,10))
proj = ccrs.Mercator()
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0,10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for name in bird_names:
    ix = birddata['bird_name'] == name
    x,y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)

plt.legend(loc="upper left")
plt.savefig("map.pdf")
plt.clf()