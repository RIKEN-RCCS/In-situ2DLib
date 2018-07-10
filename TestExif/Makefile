G=-g
PY_VER=3.6
FC=gfortran
CC=gcc
FFLAGS=-I.. $(G)
CFLAGS=-I.. $(G)
LDFLAGS=../libPi2D.a `python$(PY_VER)-config --ldflags` 

ARCH=$(shell uname -s)
ifeq ($(ARCH), Darwin)
LIBS=-lc++
else
LIBS=-lstdc++
endif

.SUFFIXES: .c .f90 .o

.f90.o:
	$(FC) $(FFLAGS) -c $<

.c.o:
	$(CC) $(CFLAGS) -c $<

TestF: TestF.o
	$(FC) -o $@ TestF.o $(LDFLAGS) $(LIBS)

TestC: TestC.o
	$(CC) -o $@ TestC.o $(LDFLAGS) $(LIBS)

clean:
	rm -fr TestC TestF *.o *.pyc __pycache__
