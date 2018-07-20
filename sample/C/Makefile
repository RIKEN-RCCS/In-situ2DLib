ARCH=$(shell uname -s)
G=-g
CC=gcc
CFLAGS=$(G) `$(PI2D_DIR)/bin/Pi2D-config --cflags`
LDFLAGS=`$(PI2D_DIR)/bin/Pi2D-config --ldflags`
ifeq ($(ARCH), Darwin)
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs` -lc++
else
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs` -lstdc++
endif

.SUFFIXES: .c .o

.c.o :
	$(CC) $(CFLAGS) -c $<

Test : Test.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *.pyc Test out*.png __pycache__
