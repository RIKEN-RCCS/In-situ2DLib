#!/bin/make
ifdef PI2D_DIR
PI2D_DIR := $(shell echo ${PI2D_DIR})
else
PI2D_DIR := $(HOME)/Pi2D
endif
ifdef PY_VER
PY_VER := $(shell echo ${PY_VER})
else
PY_VER := 3.6
endif
$(info PY_VER: $(PY_VER))
$(info PI2D_DIR: $(PI2D_DIR))

#G=-g
CXX = g++
AR = ar
RANLIB = ranlib
INSTALL = install

RFLG=
LFLG=

CXXFLAGS = $(G) $(RFLG) `python$(PY_VER)-config --cflags | sed -e 's/-arch i386//'` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS = `python$(PY_VER)-config --ldflags`
ARFLGS = crsv

PROD := libPi2D$(LFLG).a
TARGET := $(PROD) double

OBJS :=	Pi2D.o LUT.o \
	Pi2D_json.o \
	pi2d_exif.o

HEADERS := LUT.h Pi2D.h Pi2Ddefs.h pi2d_c.h picojson.h pi2d_f.inc
LICENCES := LICENSE LICENSE.picojson
DOCS := doc/SystemDesign.pdf doc/UserGuide.pdf

all : $(TARGET)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $<

$(PROD): $(OBJS)
	$(AR) $(ARFLGS) $(PROD) $(OBJS)
	$(RANLIB) $(PROD)

double:
	$(MAKE) clean
	$(MAKE) TARGET=libPi2D_d.a PI2D_DIR=$(PI2D_DIR) PY_VER=$(PY_VER) RFLG="-D_REAL_DBL" LFLG="_d"

install: install-static install-lib

install-lib : 
	$(INSTALL) -d $(PI2D_DIR)
	$(INSTALL) -d $(PI2D_DIR)/lib
	$(INSTALL) -m 644 libPi2D.a libPi2D_d.a $(PI2D_DIR)/lib

install-static :
	$(INSTALL) -d $(PI2D_DIR)
	$(INSTALL) -d $(PI2D_DIR)/bin
	$(INSTALL) -d $(PI2D_DIR)/lib
	$(INSTALL) -d $(PI2D_DIR)/include
	$(INSTALL) -d $(PI2D_DIR)/doc
	$(INSTALL) Pi2D-config $(PI2D_DIR)/bin
	$(INSTALL) -m 644 $(HEADERS) $(PI2D_DIR)/include
	$(INSTALL) -m 644 $(DOCS) $(PI2D_DIR)/doc
	$(INSTALL) -m 644 Pi2D.py $(LICENCES) $(PI2D_DIR)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) lib*.a

