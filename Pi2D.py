# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

g_fig_list = set()


def DrawS(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vt, z, lut, nlevel, cbShow, lwidth, clrPos, clrs,
          cbPos, cbSz, cbHrz, cbTic):
  global g_fig_list
  _dpi = 100
  if ( mid in g_fig_list ):
    fig = plt.figure(mid)
  else:
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)
    g_fig_list.add(mid)

  if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
    pass
  else:
    plt.axis(vp)

  nl = int(nlevel)
  if nl < 1: nl = 5

  plt.tick_params(labelbottom='off', bottom='off')
  plt.tick_params(labelleft='off', left='off')
  plt.gca().spines['right'].set_visible(False)
  plt.gca().spines['left'].set_visible(False)
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['bottom'].set_visible(False)

  _pad = -0.22 * (100.0 / _dpi)
  plt.tight_layout(pad=_pad)

  #import pdb; pdb.set_trace()

  if ( len(lut) == 0 ):
    _cmap = None
  else:
    clrlist = []
    for pos, clr in zip(clrPos, clrs):
      clrlist.append((pos, clr))
    _cmap = LinearSegmentedColormap.from_list(lut, clrlist)

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
      cont = plt.contourf(x, y, z, nl, cmap=_cmap)
    elif ( vt == 1 ):
      cont = plt.contour(x, y, z, nl, cmap=_cmap, linewidths=lwidth)
  else:
    #print("empty coord")
    if ( vt == 0 ):
      cont = plt.contourf(z, nl, cmap=_cmap)
    elif ( vt == 1 ):
      cont = plt.contour(z, nl, cmap=_cmap, linewidths=lwidth)

  if ( cbShow ):
    #pass
    cax = fig.add_axes([cbPos[0], cbPos[1], cbSz[0], cbSz[1]])
    if ( cbHrz ):
      plt.colorbar(cont, cax, orientation='horizontal')
    else:
      plt.colorbar(cont, cax, orientation='vertical')

  return True

def DrawV(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vals, vlen, vid, lut, cbShow, lwidth, vmag, vratio):
  global g_fig_list
  _dpi = 100
  if ( mid in g_fig_list ):
    fig = plt.figure(mid)
  else:
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)

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

def Output(mid, outname, step, row, col, proc):
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
  
  fig = plt.figure(mid)
  fig.savefig(fname)

  g_fig_list.remove(mid)

  return True

