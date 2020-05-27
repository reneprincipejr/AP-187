#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:54:00 2019

@author: rxander
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits import mplot3d
from skimage.restoration import unwrap_phase as unw

m = 50
n = 150
theta = [0, np.pi/2, np.pi, 3*np.pi/2]
x = y = np.linspace(0,n, 900) #1000
X,Y = np.meshgrid(x,y)

#for num in range(len(theta)):
#    Z = m*(1+np.cos(X+theta[num]))+20
#    plt.figure(facecolor = 'black')
#    plt.axis('off')
#    plt.imshow(Z, cmap = 'gray', vmin = 0, vmax=255)
#    plt.savefig(str(num)+'.png', dpi=300)

flat1 = plt.imread('flat-1.jpg')[550:1450, 1800:2700]
flat1 = cv2.cvtColor(flat1, cv2.COLOR_RGB2GRAY)
flat2 = plt.imread('flat-2.jpg')[550:1450, 1800:2700]
flat2 = cv2.cvtColor(flat2, cv2.COLOR_RGB2GRAY)
flat3 = plt.imread('flat-3.jpg')[550:1450, 1800:2700]
flat3 = cv2.cvtColor(flat3, cv2.COLOR_RGB2GRAY)
flat4 = plt.imread('flat-4.jpg')[550:1450, 1800:2700]
flat4 = cv2.cvtColor(flat4, cv2.COLOR_RGB2GRAY)

#plt.figure()
#plt.axis('off')
#plt.imshow(flat1, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('flat1.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(flat2, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('flat2.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(flat3, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('flat3.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(flat4, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('flat4.png', dpi=300)

iters = 1
for i in range(iters):
    I1 = np.mean(flat1) + flat1*np.cos(theta[0])
    I2 = np.mean(flat2) - flat2*np.sin(theta[1])
    I3 = np.mean(flat3) - flat3*np.cos(theta[2])
    I4 = np.mean(flat4) + flat4*np.sin(theta[3])
    flat1 = I1.copy()
    flat2 = I2.copy()
    flat3 = I3.copy()
    flat4 = I4.copy()
phase = np.arctan2((I4-I2), (I1-I3))
phase = phase.astype(float)

#plt.figure()
#plt.axis('off')
#plt.title('Wrapped Phase: Reference')
#plt.imshow(phase, cmap = 'gray', vmin = -np.pi, vmax = np.pi)
#plt.colorbar()
#plt.savefig('phasew.png', dpi=300)

obj1 = plt.imread('obj-1.jpg')[550:1450, 1800:2700]
obj1 = cv2.cvtColor(obj1, cv2.COLOR_RGB2GRAY)
obj2 = plt.imread('obj-2.jpg')[550:1450, 1800:2700]
obj2 = cv2.cvtColor(obj2, cv2.COLOR_RGB2GRAY)
obj3 = plt.imread('obj-3.jpg')[550:1450, 1800:2700]
obj3 = cv2.cvtColor(obj3, cv2.COLOR_RGB2GRAY)
obj4 = plt.imread('obj-4.jpg')[550:1450, 1800:2700]
obj4 = cv2.cvtColor(obj4, cv2.COLOR_RGB2GRAY)

#plt.figure()
#plt.axis('off')
#plt.imshow(obj1, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('obj1.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(obj2, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('obj2.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(obj3, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('obj3.png', dpi=300)
#plt.figure()
#plt.axis('off')
#plt.imshow(obj4, cmap = 'gray', vmin = 0, vmax = 255)
#plt.savefig('obj4.png', dpi=300)

for i in range(iters):
    I1 = np.mean(obj1) + obj1*np.cos(theta[0])
    I2 = np.mean(obj2) - obj2*np.sin(theta[1])
    I3 = np.mean(obj3) - obj3*np.cos(theta[2])
    I4 = np.mean(obj4) + obj4*np.sin(theta[3])
    obj1 = I1.copy()
    obj2 = I2.copy()
    obj3 = I3.copy()
    obj4 = I4.copy()
phase2 = np.arctan2((I4-I2), (I1-I3))
phase2 = phase2.astype(float)

#plt.figure()
#plt.axis('off')
#plt.title('Wrapped Phase: Object')
#plt.imshow(phase2, cmap = 'gray', vmin = -np.pi, vmax = np.pi)
#plt.colorbar()
#plt.savefig('phase2w.png', dpi=300)

for i in range(iters):
    phase2u = np.unwrap(phase2, axis = 1)
    phaseu = np.unwrap(phase, axis = 1)

    phase2u = np.unwrap(phase2u, axis=0)
    phaseu = np.unwrap(phaseu, axis=0)
    
#    phase2u = unw(phase2)
#    phaseu = unw(phase)
    
    phase2 = phase2u.copy()
    phase = phaseu.copy()

#plt.figure()
#plt.axis('off')
#plt.title('Unwrapped Phase: Reference')
#plt.imshow(phase, cmap = 'gray')
#plt.colorbar()
#plt.savefig('phaseu.png', dpi=300)
#
#plt.figure()
#plt.axis('off')
#plt.title('Unwrapped Phase: Object')
#plt.imshow(phase2, cmap = 'gray')
#plt.colorbar()
#plt.savefig('phase2u.png', dpi=300)

x = y = np.linspace(0, 1, 900) #1000
X,Y = np.meshgrid(x,y)
z = (phase2-phase)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,z, cmap='jet')
ax.set_title('Recovered 3D Surface')
#ax.set_zlim(-5,5)