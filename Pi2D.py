# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

g_fig_list = set()


def DrawS(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vt, z, lut, nlevel, bgClr, cbShow, lwidth, clrPos, clrs):
  """
  等高線図の描画
  """
  global g_fig_list
  try:
    _dpi = 100
    if ( mid in g_fig_list ):
      fig = plt.figure(mid)
    else:
      x = imgSz[0] / _dpi
      y = imgSz[1] / _dpi
      fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)
      ax = fig.add_subplot(111)
      ax.patch.set_facecolor(bgClr)
      g_fig_list.add(mid)

    if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
      pass
    else:
      plt.axis(vp)

    plt.tick_params(labelbottom=False, bottom=False)
    plt.tick_params(labelleft=False, left=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    _pad = -0.22 * (100.0 / _dpi)
    plt.tight_layout(pad=_pad)

    _cmap = None
    _norm = None
    _colors = None
    if ( len(lut) != 0 ):
      if ( len(clrPos) == 1 ):
        _colors = clrs
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
        cont = plt.contourf(x, y, z, nL, cmap=_cmap, norm=_norm,
                            colors=_colors)
      elif ( vt == 1 ):
        cont = plt.contour(x, y, z, nL, cmap=_cmap, norm=_norm,
                           colors=_colors, linewidths=lwidth)
    else:    # empty coord
      if ( vt == 0 ):
        cont = plt.contourf(z, nL, cmap=_cmap, norm=_norm, colors=_colors)
      elif ( vt == 1 ):
        cont = plt.contour(z, nL, cmap=_cmap, norm=_norm, colors=_colors,
                           linewidths=lwidth)
  except:
    return False

  return True

def DrawV(mid, imgSz, vp, arrSz, coord, veclen, vecid,
          vals, vlen, vid, lut, bgClr, cbShow, lwidth, vmag, vratio,
          clist, clrPos, clrs):
  """
  ベクトル図の描画
  """
  global g_fig_list
  try:
    _dpi = 100.0
    ax = None

    if ( mid in g_fig_list ):
      fig = plt.figure(mid)
    else:
      x = imgSz[0] / _dpi
      y = imgSz[1] / _dpi
      fig = plt.figure(mid, figsize=(x, y), dpi=_dpi)
      ax = fig.add_subplot(111)
      ax.patch.set_facecolor(bgClr)
      g_fig_list.add(mid)

    if ( vp[0] == 0.0 and vp[1] == 0.0 and vp[2] == 0.0 and vp[3] == 0.0 ):
      pass
    else:
      plt.axis(vp)

    plt.tick_params(labelbottom=False, bottom=False)
    plt.tick_params(labelleft=False, left=False)
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
    d = np.linalg.norm(imgSz-np.zeros(2))
    _width = lwidth / d
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
                 scale=_scale, width=_width, headwidth=wid, headlength=leng)
    else:
      plt.quiver(u, v, angles='xy', scale_units='xy', color=clist,
                 scale=_scale, width=_width, headwidth=wid, headlength=leng)
  except:
    return False

  return True

def DrawCB(mid, lut, clrPos, clrs, cbSz, cbPos, cbHrz, cbTic, cbTicClr):
  """
  カラーバーの描画
  """
  global g_fig_list
  try:
    if ( mid in g_fig_list ):
      fig = plt.figure(mid)
    else:
      return False

    _cmap = None
    _norm = None
    _ticks = None
    if ( len(lut) != 0 ):
      cmin = min(clrPos)
      cmax = max(clrPos)
      crange = cmax - cmin
      cposList = [cmin]
      clrarr = clrs
      if ( crange != 0.0 ):
        cposList = [((v - cmin) / crange) for v in clrPos]
      else:
        cposList.append(cmin+1.0)
        clrarr = [clrs[0], clrs[0]]
      clrlist = []
      for cpos, clr in zip(cposList, clrarr):
        clrlist.append((cpos, clr))
      _cmap = LinearSegmentedColormap.from_list(lut, clrlist)
      _norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
      _ticks = [cmin]
      df = crange / (cbTic - 1)
      for i in range(cbTic - 1):
        v = cmin + df * (i + 1)
        _ticks.append(v)

    cax = fig.add_axes([cbPos[0], cbPos[1], cbSz[0], cbSz[1]])
    if ( cbHrz ):
      cb = mpl.colorbar.ColorbarBase(cax, cmap=_cmap, norm=_norm, ticks=_ticks,
                                     orientation='horizontal')
      cb.ax.tick_params(colors=(cbTicClr))
    else:
      cb = mpl.colorbar.ColorbarBase(cax, cmap=_cmap, norm=_norm, ticks=_ticks,
                                     orientation='vertical')
      cb.ax.tick_params(colors=(cbTicClr))
  except:
    return False

  return True

def Output(mid, outname, step, row, col, proc):
  """
  画像の出力
  """
  global g_fig_list
  try:
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
  
    if ( mid in g_fig_list ):
      fig = plt.figure(mid)
    else:
      return False

    fig.savefig(fname)

    plt.close(fig)
    g_fig_list.remove(mid)
  except:
    return False

  return True

