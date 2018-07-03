#!/bin/make
prefix=`echo $$HOME`
PI2D_DIR=$(prefix)/Pi2D

G=-g
CXX = g++
AR = ar
RANLIB = ranlib
INSTALL = install

PY_VER = 3.6

CXXFLAGS = $(G) `python$(PY_VER)-config --cflags | sed -e 's/-arch i386//'` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS = `python$(PY_VER)-config --ldflags`
ARFLGS = crsv

PROG = Pi2D
TARGET = lib$(PROG).a

OBJS =	Pi2D.o LUT.o \
	Pi2D_json.o \
	pi2d_exif.o

HEADERS = LUT.h  Pi2D.h  Pi2Ddefs.h  pi2d_c.h  picojson.h pi2d_f.inc
LICENCES = LICENSE  LICENSE.picojson

all : $(OBJS) $(TARGET)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $<

$(TARGET): $(OBJS)
	$(AR) $(ARFLGS) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

install : $(TARGET)
	$(INSTALL) -d $(PI2D_DIR)
	$(INSTALL) -d $(PI2D_DIR)/bin
	$(INSTALL) -d $(PI2D_DIR)/lib
	$(INSTALL) -d $(PI2D_DIR)/include
	$(INSTALL) -t $(PI2D_DIR)/include $(HEADERS)
	$(INSTALL) -t $(PI2D_DIR)/lib $(TARGET)
	$(INSTALL) -t $(PI2D_DIR)/bin Pi2D-config
	$(INSTALL) -m 644 -t $(PI2D_DIR) Pi2D.py $(LICENCES)

clean:
	$(RM) $(OBJS) $(TARGET)

