G=-g
FC=gfortran
CC=gcc
FFLAGS=-I.. $(G)
CFLAGS=-I.. $(G)
LIBS=../libPi2D.a `python3.5-config --ldflags` -lstdc++

.SUFFIXES: .c .f90 .o

.f90.o:
	$(FC) $(FFLAGS) -c $<

.c.o:
	$(CC) $(CFLAGS) -c $<

TestF: TestF.o
	$(FC) -o $@ TestF.o $(LIBS)

TestC: TestC.o
	$(CC) -o $@ TestC.o $(LIBS)

clean:
	rm -fr TestC TestF *.o *.pyc __pycache__
