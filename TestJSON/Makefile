G=-g
PY_VER=3.6
CXX=g++
CXXFLAGS=$(G) -I.. `python$(PY_VER)-config --includes`
LDFLAGS=../libPi2D.a `python$(PY_VER)-config --ldflags`

.SUFFIXES: .cpp .o

TestExport : TestExport.o ../libPi2D.a
	$(CXX) -o $@ $< $(LDFLAGS)

TestImport : TestImport.o ../libPi2D.a
	$(CXX) -o $@ $< $(LDFLAGS)

Y.json : X.json
	cat X.json | jq '.' > Y.json

clean:
	rm -rf *.o *.pyc X.json Y.json TestExport
