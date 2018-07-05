# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

g_fig_list = set()


def DrawS(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vt, z, lut, nlevel, cbShow, lwidth, clrPos, clrs,
          cbSz, cbPos, cbHrz, cbTic):
  global g_fig_list
  _dpi = 100
  if ( mid in g_fig_list ):
    fig = plt.figure(mid)
  else:
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)
    # set as no margin
    #_pad = -0.22 * (100.0 / _dpi)
    #plt.tight_layout(pad=_pad)
    # add id to global set variable
    g_fig_list.add(mid)

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

  #import pdb; pdb.set_trace()

  if ( len(lut) == 0 ):
    _cmap = None
    _norm = None
  else:
    cmin = min(clrPos)
    cmax = max(clrPos)
    crange = cmax - cmin
    cposList = [0.0]
    if ( crange != 0.0 ):
      cposList = [((v - cmin) / crange) for v in clrPos]
    clrlist = []
    #for pos, clr in zip(clrPos, clrs):
    for cpos, clr in zip(cposList, clrs):
      clrlist.append((cpos, clr))
    _cmap = LinearSegmentedColormap.from_list(lut, clrlist)
    _norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

  if nlevel <= 0:
    nL = 10
  else:
    nL = int(nlevel)
    
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
      cont = plt.contourf(x, y, z, nL, cmap=_cmap, norm=_norm)
    elif ( vt == 1 ):
      cont = plt.contour(x, y, z, nL, cmap=_cmap, norm=_norm,
                         linewidths=lwidth)
  else:    # empty coord
    if ( vt == 0 ):
      cont = plt.contourf(z, nL, cmap=_cmap, norm=_norm)
    elif ( vt == 1 ):
      cont = plt.contour(z, nL, cmap=_cmap, norm=_norm,
                         linewidths=lwidth)

  if ( cbShow ):
    cax = fig.add_axes([cbPos[0], cbPos[1], cbSz[0], cbSz[1]])
    if ( cbHrz ):
      plt.colorbar(cont, cax, orientation='horizontal')
    else:
      plt.colorbar(cont, cax, orientation='vertical')

  return True

def DrawV(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vals, vlen, vid, lut, cbShow, lwidth, vmag, vratio,
          clist, clrPos, clrs, cbSz, cbPos, cbHrz, cbTi):
  #import pdb; pdb.set_trace()

  global g_fig_list
  _dpi = 100
  if ( mid in g_fig_list ):
    fig = plt.figure(mid)
  else:
    x = imgSz[0] / 100.0
    y = imgSz[1] / 100.0
    fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)
    # set as no margin
    #_pad = -0.22 * (100.0 / _dpi)
    #plt.tight_layout(pad=_pad)
    # add id to global set variable
    g_fig_list.add(mid)

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
    _norm = None
  else:
    cmin = min(clrPos)
    cmax = max(clrPos)
    crange = cmax - cmin
    cposList = [0.0]
    if ( crange != 0.0 ):
      cposList = [((v - cmin) / crange) for v in clrPos]
    clrlist = []
    for cpos, clr in zip(cposList, clrs):
      clrlist.append((cpos, clr))
    _cmap = LinearSegmentedColormap.from_list(lut, clrlist)
    _norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

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
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', color=clist,
               scale=_scale, headwidth=wid, headlength=leng)
  else:
    plt.quiver(u, v, angles='xy', scale_units='xy',
               scale=_scale, headwidth=wid, headlength=leng)

  if ( cbShow ):
    #pass
    cax = fig.add_axes([cbPos[0], cbPos[1], cbSz[0], cbSz[1]])
    #_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    if ( cbHrz ):
      #plt.colorbar(cont, cax, orientation='horizontal')
      mpl.colorbar.ColorbarBase(cax, cmap=_cmap, norm=_norm,
      #mpl.colorbar.ColorbarBase(cax, cmap=_cmap,
                                orientation='horizontal')
    else:
      #plt.colorbar(cont, cax, orientation='vertical')
      mpl.colorbar.ColorbarBase(cax, cmap=_cmap, norm=_norm,
      #mpl.colorbar.ColorbarBase(cax, cmap=_cmap,
                                orientation='vertical')

  return True

def Output(mid, outname, step, row, col, proc):
  fname = outname
  p = fname.find('%S')
  while ( p != -1 ):
    n = int(fname[p+2])
    if ( n != -1 ):
      s = '%0' + fname[p+2] + 'd'
      s = s % step
      fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%S')
  p = fname.find('%R')
  while ( p != -1 ):
    n = int(fname[p+2])
    if ( n != -1 ):
      s = '%0' + fname[p+2] + 'd'
      s = s % row
      fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%R')
  p = fname.find('%C')
  while ( p != -1 ):
    n = int(fname[p+2])
    if ( n != -1 ):
      s = '%0' + fname[p+2] + 'd'
      s = s % col
      fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%C')
  p = fname.find('%P')
  while ( p != -1 ):
    n = int(fname[p+2])
    if ( n != -1 ):
      s = '%0' + fname[p+2] + 'd'
      s = s % proc
      fname = fname[0:p] + s + fname[p+3:]
    p = fname.find('%P')
  
  fig = plt.figure(mid)
  fig.savefig(fname)

  g_fig_list.remove(mid)

  return True

