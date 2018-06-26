import sys, os
import MicroEnv as me
from mpi4py import MPI

def FUNC():
  hwmess = 'process %d of %d on %s.'
  myrank = MPI.COMM_WORLD.Get_rank()
  nprocs = MPI.COMM_WORLD.Get_size()
  procnm = MPI.Get_processor_name()
  print (hwmess % (myrank, nprocs, procnm))

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

