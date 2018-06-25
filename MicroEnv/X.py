import sys, os
import MicroEnv as me

def FUNC():
  uvw = me.getArray('uvw');
  uvw = uvw * 10.0
  ret = me.setArray('uvw', uvw)

  idx = me.getArray('idx')
  idx = idx + 50
  ret = me.setArray('idx', idx)

  rho = me.getArray('rho')
  rho[5][5][5] = 10.0
  ret = me.setArray('rho', rho)

  return 0

