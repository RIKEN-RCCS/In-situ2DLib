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

  return 0

