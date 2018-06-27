G=-g
PY_VER=3.6
CXX=g++
CXXFLAGS=$(G) -I.. `python$(PY_VER)-config --cflags`
LDFLAGS=../libPi2D.a `python$(PY_VER)-config --ldflags`

.SUFFIXES: .cpp .o

TestExport : TestExport.o
	$(CXX) -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o *.pyc X.json TestExport
