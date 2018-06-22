import sys, os
import MicroEnv as me

def FUNC():
  uvw = me.getArray('uvw');

  uvw = uvw * 10.0

  me.setArray('uvw', uvw)
  
  return 0

