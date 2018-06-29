# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Pi2D:

  def __init__(self):
    print("python done")

  def DrawS(self, mid, imgSz, vp, arrSz, coord, veclen, vecid,
            vt, z, lut, nlevel, cbShow, lwidth, clrPos, clrs,
            cbPos, cbSz, cbHrz, cbTic):
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
      clrlist = []
      for pos, clr in zip(clrPos, clrs):
        clrlist.append((pos, clr))
      _cmap = LinearSegmentedColormap.from_list(lut, clrlist)

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
            vals, vlen, vid, lut, cbShow, lwidth, vmag, vratio):
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

    csz = arrSz[0] * arrSz[1];
    val0 = vals.reshape(csz, vlen)
    u0 = val0[:, [vid[0]]]
    u1 = u0.flatten()
    u = u1.reshape(arrSz[0], arrSz[1])
    v0 = val0[:, [vid[1]]]
    v1 = v0.flatten()
    v = v1.reshape(arrSz[0], arrSz[1])

    _scale = 1.0 / vmag
    wid = vratio[0]
    leng = vratio[1]
    if ( vratio[0] == -1 ):
      wid = 3.0
    if ( vratio[1] == -1 ):
      leng = 5.0

    if ( len(coord) != 1 ):
      coord0 = coord.reshape(csz, veclen)
      x0 = coord0[:, [vecid[0]]]
      x1 = x0.flatten()
      x = x1.reshape(arrSz[0], arrSz[1])
      y0 = coord0[:, [vecid[1]]]
      y1 = y0.flatten()
      y = y1.reshape(arrSz[0], arrSz[1])
      #plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=clist)
      plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                 scale=_scale, headwidth=wid, headlength=leng)
    else:
      plt.quiver(u, v, angles='xy', scale_units='xy',
                 scale=_scale, headwidth=wid, headlength=leng)

    return True

  def Output(self, outname, step, row, col, proc):
    fname = outname
    p = fname.find('%S')
    if ( p != -1 ):
      n = int(fname[p+2])
      if ( n != -1 ):
        s = '%0' + fname[p+2] + 'd'
        s = s % step
        fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%R')
    if ( p != -1 ):
      n = int(fname[p+2])
      if ( n != -1 ):
        s = '%0' + fname[p+2] + 'd'
        s = s % row
        fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%C')
    if ( p != -1 ):
      n = int(fname[p+2])
      if ( n != -1 ):
        s = '%0' + fname[p+2] + 'd'
        s = s % col
        fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%P')
    if ( p != -1 ):
      n = int(fname[p+2])
      if ( n != -1 ):
        s = '%0' + fname[p+2] + 'd'
        s = s % proc
        fname = fname[0:p] + s + fname[p+3:]
    
    self.fig.savefig(fname)
    return True

