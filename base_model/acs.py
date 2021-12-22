import sys
from math import sqrt
import json

def acs( batchmeansList, b ):

    K = 50                                  # K is the maximum lag */
    SIZE = (K + 1)


    i = 0                                   # data point index              */
    sum = 0.0                               # sums x[i]                     */
    j = 0                                   # lag index                     */
    hold = []                               # K + 1 most recent data points */
    p = 0                                   # points to the head of 'hold'  */
    cosum = [0 for i in range(0,SIZE)]      # cosum[j] sums x[i] * x[i+j]   */


    while (i < SIZE):                        # initialize the hold array with */
      x = float( batchmeansList.pop(0) )     # the first K + 1 data values    */
      sum += x
      hold.append(x)
      i += 1
    #EndWhile

    if len(batchmeansList) != 0: x = batchmeansList.pop(0)   

    while (x):
      for j in range(0,SIZE):
        cosum[j] += hold[p] * hold[(p + j) % SIZE]
      x = float(x) #lines read in as string
      sum    += x
      hold[p] = x
      p       = (p + 1) % SIZE
      i += 1 
      if len(batchmeansList) != 0: x = batchmeansList.pop(0)   
      else: x = 0
    #EndWhile
    n = i #the total number of data points

    while (i < n + SIZE):         # empty the circular array */
      for j in range(0,SIZE):
        cosum[j] += hold[p] * hold[(p + j) % SIZE]
      hold[p] = 0.0
      p       = (p + 1) % SIZE
      i += 1 
    #EndWhile

    mean = sum / n
    for j in range(0,K+1):  
      cosum[j] = (cosum[j] / (n - j)) - (mean * mean)
    

    autocorrelations = [ (c / cosum[0]) for c in cosum ]

    with open( "acs.json", 'r' ) as f:
      data = json.load(f)
      data[b] = autocorrelations
      f.close()
    
    with open( "acs.json", 'w' ) as f:
      json.dump( data, f )
      f.close()


