#!/bin/make
CXX = g++
AR = ar

PY_VER = 3.6

CXXFLAGS = `python$(PY_VER)-config --cflags` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS = `python$(PY_VER)-config --ldflags`
ARFLGS = crsv

PROG = Pi2D
TARGET = lib$(PROG).a

OBJS = Pi2D.o LUT.o

all : $(OBJS) $(TARGET)

.cpp.o :
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $<

$(TARGET):	$(OBJS)
	$(AR) $(ARFLGS) $(TARGET) $(OBJS)

clean:
	$(RM) $(OBJS) $(TARGET)
