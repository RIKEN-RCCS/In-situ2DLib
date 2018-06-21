import sys, os

import emb

def FUNC(d):
  a = d['a']
  a[1][1] = 1
  sys.stdout.write(str(a) + '\n');

  sys.stdout.write("Number of arguments=%d\n" % emb.numargs())
  return 0

