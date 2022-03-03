# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:46:10 2018

@author: Administrator
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


plt.axis([0, 577, 0, 0.15])

red_patch = mpatches.Patch(color='red', label='training error rate')
blue_patch = mpatches.Patch(color='blue', label='test error rate')

plt.ylabel('RMS error', fontsize = 14)
plt.xlabel('training set size', fontsize = 14)
plt.title('learning curves', fontsize = 14, y = 1.03)

plt.legend(handles=[red_patch,blue_patch])

a1=0.073
b1=0.073
a2=0.0567
b2=0.0567
a3=0.0549
b3=0.0548
a4=0.05
b4=0.05
a5=0.0438
b5=0.0438



plt.plot([40,100,200,400,576], [a1,a2,a3,a4,a5],'r-', lw=2)
plt.plot([40,100,200,400,576], [b1,b2,b3,b4,b5],'b-', lw=2)


plt.plot(40, a1, "o",color='r')
plt.plot(40, b1, "o",color='b')

plt.plot(100, a2, "o",color='r')
plt.plot(100, b2, "o",color='b')

plt.plot(200, a3, "o",color='r')
plt.plot(200, b3, "o",color='b')

plt.plot(400, a4, "o",color='r')
plt.plot(400, b4, "o",color='b')

plt.plot(576, a5, "o",color='r')
plt.plot(576, b5, "o",color='b')



plt.show()

