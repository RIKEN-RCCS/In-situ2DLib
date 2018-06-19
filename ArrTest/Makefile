PY_VER=3.4
CXX=g++
G=-g -O0
FLAGS = `python$(PY_VER)-config --cflags --ldflags` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`

all : DRAWTEST

DRAWTEST: main.cpp
	$(CXX) $(G) -o $@ $< $(FLAGS)

