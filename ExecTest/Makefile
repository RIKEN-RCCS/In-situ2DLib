G=-g
PY_VER=3.6
CXX=g++
CXXFLAGS=$(G) -I.. `python$(PY_VER)-config --includes` \
	-I`python$(PY_VER) -c 'import numpy, sys; sys.stdout.write(numpy.get_include())'`
LDFLAGS=../libPi2D.a `python$(PY_VER)-config --ldflags`

.SUFFIXES: .cpp .o

run : run.o ../libPi2D.a
	$(CXX) -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o *.pyc run out*.png __pycache__
