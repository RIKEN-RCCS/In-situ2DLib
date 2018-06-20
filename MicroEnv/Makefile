PY_VER=3.6
CXX=g++
G=-g -O0
CXXFLAGS = `python$(PY_VER)-config --cflags` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS=`python$(PY_VER)-config --ldflags`

TARGET=Test
OBJS = Test.o MicroEnv.o

all : Test

$(TARGET): $(OBJS)
	$(CXX) $(G) -o $@ $(OBJS) $(LDFLAGS)

clean :
	rm -fr $(TARGET) $(OBJS) *.pyc *.pyo __pycache__

