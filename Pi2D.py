# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Pi2D:

  def __init__(self):
    print("python done")

  def DrawS(self, mid, imgSz, vp, vt, z):
    _dpi = 100
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    self.fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)

    if ( vp[0] != 0 and vp[1] != 0 and vp[2] != 0 and vp[3] != 0 ):
      plt.axis(vp)

    plt.tick_params(labelbottom='off', bottom='off')
    plt.tick_params(labelleft='off', left='off')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    _pad = -0.22 * (100.0 / _dpi)
    plt.tight_layout(pad=_pad)

    if ( vt == 0 ):
      plt.contourf(z)
    elif ( vt == 1 ):
      plt.contour(z)

    return True

  def DrawV(self):
    return True

  def Output(self, outname):
    #outname = "out.png"
    print(outname)
    self.fig.savefig(outname)
    return True

