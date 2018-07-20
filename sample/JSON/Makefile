G=-g
CXX=g++
CXXFLAGS=$(G) `$(PI2D_DIR)/bin/Pi2D-config --cflags`
LDFLAGS=`$(PI2D_DIR)/bin/Pi2D-config --ldflags`
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs`
MAKE=make
JQ=jq '.'
TARGET=TestExport TestImport

.SUFFIXES: .cpp .o

all : $(TARGET)

TestExport : TestExport.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

TestImport : TestImport.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

Y.json : X.json
	cat X.json | $(JQ) > Y.json

test: $(TARGET)
	./TestExport
	$(MAKE) Y.json
	./TestImport

clean:
	rm -rf *.o *.pyc X.json Y.json $(TARGET)
