#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:21:19 2017

"""

import os #Used in file saving function
import numpy as np
from numba import jit, vectorize, cuda
import sys, math, cmath, time
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from gratingLib import *
from cudaKernels import *
from time import gmtime, strftime




#Define initial parameters #################################################################################
screen_distance = 5e7 #nm
screen_length = 1e7
second_grating_distance = 5e7 #nm
wavelength = .56 #nm
U_0 = 1 #?
wavenumber = 2 * np.pi / wavelength
numOfSlits = 200 # number of slits in each grating
numOfPointSources = 100  # number of point sources in each slit
numObsPoints = 1000    # number of observing points on the screen
spacingType = 'uniform'
slitLength = 50 #nm
newSimulation = False
runNum = 1 #Used to dynamically name files. Change every time you run a simulation. Otherwise it will write
            # over old data
timings = []
############################################################################################################

# Observing screen size
#center of screen will automatically be at 0.5e7 nm
# Change this based on size of gratings
screenStart = 0e7
screenEnd = 1e7


#create array of positions that represent an observing screen
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
observingPositions = np.linspace(screenStart,screenEnd,numObsPoints)
# Build gratings and fill with point sources
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
firstGrating = Grating(x=0, length=screen_length, numberOfSlits=numOfSlits, slitWidth=slitLength, sourcesPerSlit = numOfPointSources)
# Build second grating and fill with point sources
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
secondGrating = Grating(x=second_grating_distance, length=screen_length, numberOfSlits=numOfSlits, slitWidth=slitLength, sourcesPerSlit = numOfPointSources)
# Define initial source
# Options are 'spherical' and 'plane'
# Initial source position is -(distance from first grating in nm)
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
initSource = InitialSource(xPosition= -1e7, yPosition=screen_length/2, waveType='plane', initialAmplitude=1.0)

# generate source amplitudes and phases based on the initial source and the first gratings point source positions
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
sourceAmps, sourcePhase = initSource.propogate(firstGrating.x, firstGrating.pointSourcePositions, wavenumber, normalize=True)

# add these amplitudes to the first grating's point sources
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
firstGrating.addAmplitudes(sourceAmps, sourcePhase)
#calculate information from firstGrating propagating to secondGrating
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
intensities, amplitudes, phases = intensityCalculations(screen_distance, wavenumber, firstGrating.pointSourcePositions, secondGrating.pointSourcePositions, firstGrating.pointSourceAmplitudes, firstGrating.pointSourcePhases)
#add necessary results to secondGrating's point sources
#print('Populating grating 2\n')
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
secondGrating.addAmplitudes(amplitudes, phases)

#calculate information from secondGrating propagation to observingPositions
#print('Grating 2 to Screen:\n')
timings.append(strftime("%Y/%m/%d %H:%M:%S"))
intensities2, amplitudes2, phases2 = intensityCalculations(screen_distance, wavenumber, secondGrating.pointSourcePositions, observingPositions, secondGrating.pointSourceAmplitudes, secondGrating.pointSourcePhases)
timings.append(strftime("%Y/%m/%d %H:%M:%S"))

if newSimulation:     
    with open("onSecondGratingResults_%s_run00%s.txt" %(initSource.waveType,runNum), 'w') as f:
        f.write("#source wave type: %s, time taken: %s\n" %(initSource.waveType,tf1-t01))
        f.write("#Intensity\tAmplitudes\t\tPhase\t\t\t\tPosition\n")
        for i, a, p, o in zip(intensities, amplitudes, phases, secondGrating.pointSourcePositions):
            f.write("%s\t%s\t%s\t%s\n" %(i, a, p, o))
            
    with open("onScreenResults_%s_run00%s.txt" %(initSource.waveType,runNum), 'w') as f:
        f.write("#source wave type: %s, time taken: %s" %(initSource.waveType,tf2-t02))
        f.write("#Intensity\tAmplitudes\t\tPhase\t\t\t\tPosition\n")
        for i, a, p, o in zip(intensities2, amplitudes2, phases2, observingPositions):
            f.write("%s\t%s\t%s\t%s\n" %(i, a, p, o))
        
cuda.close()


print("Function, Timestamp")
print("observingPositions," + timings[0])
print("firstGrating," + timings[1])
print("secondGrating," + timings[2])
print("initSource," + timings[3])
print("initSource.propogate," + timings[4])
print("firstGrating.addAmplitudes," + timings[5])
print("calcIntensitiesCUDA," + timings[6])
print("secondGrating.addAmplitudes," + timings[7])
print("calcIntensitiesCUDA," + timings[8])
print("End of program," + timings[9] + "\n")



# quickly plot data to see if results are reasonable
plt.figure(figsize=(15,8))
plt.plot(firstGrating.pointSourcePositions,firstGrating.pointSourceAmplitudes,'.r')
plt.savefig('dottedProfile.png', transparent=True)
plt.xlabel('Position on First Grating (nm)', fontsize = 25)
plt.ylabel('Amplitude', fontsize = 25)
plt.title('Incident on First Grating', fontsize = 30)
plt.show()


plt.figure(figsize=(15,8))
plt.plot(secondGrating.pointSourcePositions,intensities,'.r')
plt.savefig('dottedProfile.png', transparent=True)
plt.xlabel('Position on Second Grating (nm)', fontsize = 25)
plt.ylabel('Normalized Intensity', fontsize = 25)
plt.title('Incident on Second Grating', fontsize = 30)
plt.show()

maxIntensities2 = max(intensities2)
intensities2 = [i/maxIntensities2 for i in intensities2]

obsPositionsMicrons = [i/1000 for i in observingPositions]

plt.figure(figsize=(15,8))
plt.plot(obsPositionsMicrons,intensities2,'r')
plt.savefig('dottedProfile.png', transparent=True)
plt.xlabel('Position on Observing Screen (nm)', fontsize = 25)
plt.ylabel('Normalized Intensity', fontsize = 25)
plt.title('Uniform Grating', fontsize = 30)
plt.show()
