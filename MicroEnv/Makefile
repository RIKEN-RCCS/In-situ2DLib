PY_VER=3.6
CXX=g++
G=-g -O0
CXXFLAGS = `python$(PY_VER)-config --cflags` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS=`python$(PY_VER)-config --ldflags`

TARGET=Test
TARGET2=Test2
TARGET3=Test3
OBJS = Test.o MicroEnv.o
OBJS2 = Test2.o MicroEnv.o
OBJS3 = Test3.o MicroEnv.o

all : Test

$(TARGET): $(OBJS)
	$(CXX) $(G) -o $@ $(OBJS) $(LDFLAGS)

$(TARGET2): $(OBJS2)
	$(CXX) $(G) -o $@ $(OBJS2) $(LDFLAGS)

$(TARGET3): $(OBJS3)
	$(CXX) $(G) -o $@ $(OBJS3) $(LDFLAGS)

clean :
	rm -fr $(TARGET) $(OBJS) *.pyc *.pyo __pycache__

