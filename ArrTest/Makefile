CXX=g++
G=-g -O0
FLAGS = `python3.5-config --cflags --ldflags`

all : DRAWTEST

DRAWTEST: main.cpp
	$(CXX) $(G) -o $@ $< $(FLAGS)

