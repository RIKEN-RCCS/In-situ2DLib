ARCH=$(shell uname -s)
G=-g
FC=gfortran
FFLAGS=$(G) `$(PI2D_DIR)/bin/Pi2D-config --cflags` -cpp
LDFLAGS=`$(PI2D_DIR)/bin/Pi2D-config --ldflags`
ifeq ($(ARCH), Darwin)
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs` -lc++
else
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs` -lstdc++
endif

.SUFFIXES: .f90 .o

.f90.o:
	$(FC) $(FFLAGS) -c $<

Test : Test.o
	$(FC) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *.pyc Test out*.png __pycache__
