PY_VER=3.6
CXX=mpicxx
G=-g -O0
CXXFLAGS = `python$(PY_VER)-config --cflags | sed -e 's/-arch i386//'` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS=`python$(PY_VER)-config --ldflags`

TARGET=Test
TARGET2=StatTest
TARGET3=MPI4PyTest
OBJS = Test.o MicroEnv.o
OBJS2 = StatTest.o 
OBJS3 = MPI4PyTest.o 

all : Test

$(TARGET): $(OBJS)
	$(CXX) $(G) -o $@ $(OBJS) $(LDFLAGS)

$(TARGET2): $(OBJS2)
	$(CXX) $(G) -o $@ $(OBJS2) $(LDFLAGS)

$(TARGET3): $(OBJS3)
	$(CXX) $(G) -o $@ $(OBJS3) $(LDFLAGS)

clean :
	rm -fr *.o *.pyc *.pyo __pycache__ *.dSYM
	rm -f $(TARGET) $(TARGET2) $(TARGET3)

