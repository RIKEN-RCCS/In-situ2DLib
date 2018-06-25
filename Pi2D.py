# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Pi2D:

  def __init__(self):
    print("python done")

  def DrawS(self, mid, imgSz, vp, arrSz, coord, veclen, vecid,
            vt, z):
    #        vt, nlevel, z):
    _dpi = 100
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    self.fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)

    if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
      print(vp)
    else:
      plt.axis(vp)

    plt.tick_params(labelbottom='off', bottom='off')
    plt.tick_params(labelleft='off', left='off')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    _pad = -0.22 * (100.0 / _dpi)
    plt.tight_layout(pad=_pad)

    if ( len(coord) != 1 ):
      csz = arrSz[0] * arrSz[1];
      coord0 = coord.reshape(csz, veclen)
      x0 = coord0[:, [vecid[0]]]
      x1 = x0.flatten()
      x = x1.reshape(arrSz[0], arrSz[1])
      #print(x)
      y0 = coord0[:, [vecid[1]]]
      y1 = y0.flatten()
      y = y1.reshape(arrSz[0], arrSz[1])
      #print(y)
      if ( vt == 0 ):
        #plt.contourf(x, y, z, nlevel)
        plt.contourf(x, y, z)
      elif ( vt == 1 ):
        #plt.contour(x, y, z, nlevel)
        plt.contour(x, y, z)
    else:
      #print("empty coord")
      if ( vt == 0 ):
        plt.contourf(z)
      elif ( vt == 1 ):
        plt.contour(z)
    print("dbg")

    return True

  def DrawV(self):
    return True

  def Output(self, outname):
    #outname = "out.png"
    print(outname)
    self.fig.savefig(outname)
    return True

