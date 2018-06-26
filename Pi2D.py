# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Pi2D:

  def __init__(self):
    print("python done")

  def DrawS(self, mid, imgSz, vp, arrSz, coord, veclen, vecid,
            vt, z, lut, nlevel, cbShow, lwidth):
    _dpi = 100
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    self.fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)

    if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
      pass
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

    if ( len(lut) == 0 ):
      _cmap = None
    else:
      #_cmap = lut
      _cmap = None

    if ( cbShow ):
      pass

    if ( len(coord) != 1 ):
      csz = arrSz[0] * arrSz[1];
      coord0 = coord.reshape(csz, veclen)
      x0 = coord0[:, [vecid[0]]]
      x1 = x0.flatten()
      x = x1.reshape(arrSz[0], arrSz[1])
      y0 = coord0[:, [vecid[1]]]
      y1 = y0.flatten()
      y = y1.reshape(arrSz[0], arrSz[1])
      if ( vt == 0 ):
        plt.contourf(x, y, z, nlevel, cmap=_cmap)
      elif ( vt == 1 ):
        plt.contour(x, y, z, nlevel, cmap=_cmap, linewidths=lwidth)
    else:
      #print("empty coord")
      if ( vt == 0 ):
        plt.contourf(z, nlevel, cmap=_cmap)
      elif ( vt == 1 ):
        plt.contour(z, nlevel, cmap=_cmap, linewidths=lwidth)

    return True

  def DrawV(self, mid, imgSz, vp, arrSz, coord, veclen, vecid,
            vals, vlen, vid, lut, cbShow, lwidth):
    _dpi = 100
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    self.fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)

    if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
      pass
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

    if ( len(lut) == 0 ):
      _cmap = None
    else:
      #_cmap = lut
      _cmap = None

    if ( cbShow ):
      pass

    if ( len(coord) != 1 ):
      csz = arrSz[0] * arrSz[1];
      coord0 = coord.reshape(csz, veclen)
      x0 = coord0[:, [vecid[0]]]
      x1 = x0.flatten()
      x = x1.reshape(arrSz[0], arrSz[1])
      y0 = coord0[:, [vecid[1]]]
      y1 = y0.flatten()
      y = y1.reshape(arrSz[0], arrSz[1])
      plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=clist)
    else:
      plt.quiver(u, v, angles='xy', scale_units='xy', scale=1, color=clist)

    return True

  def Output(self, outname):
    #outname = "out.png"
    print(outname)
    self.fig.savefig(outname)
    return True

