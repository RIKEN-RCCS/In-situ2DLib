import sys, os
import numpy as np

def FUNC(d):
  a = d['a']
  a[1][1] = 1
  sys.stdout.write(str(a) + '\n');

  import emb
  A = emb.getA();
  nA = len(A)
  sys.stdout.write("len(A)=%d\n" % nA)
  sys.stdout.write("A=" + str(A) +'\n')

  A2 = np.array(A, dtype=np.float64)
  for i in range(nA):
    A2[i] = i * 2.0
  sys.stdout.write("A2=" + str(A2) +'\n')
  emb.setA(A2);
  
  return 0

