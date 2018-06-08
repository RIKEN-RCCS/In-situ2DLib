#!/bin/make
CXX = g++
AR = ar
#CXXFLAGS = -O2 -Wall -fPIC
#LDFLAGS = -I/usr/include/python3.6m -L/usr/lib64 -lpython3.6m
CXXFLAGS = `/usr/bin/python3.6-config --cflags`
LDFLAGS = `/usr/bin/python3.6-config --ldflags`
ARFLGS = crsv

PROG = Pi2D
TARGET = lib$(PROG).a

OBJS = Pi2D.o

all : $(OBJS) $(TARGET)

.cpp.o :
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $<

$(TARGET):	$(OBJS)
	$(AR) $(ARFLGS) $(TARGET) $(OBJS)

clean:
	$(RM) $(OBJS) $(TARGET)
