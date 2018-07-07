G=-g
CXX=g++
CXXFLAGS=$(G) `$(PI2D_DIR)/bin/Pi2D-config --cflags`
LDFLAGS=`$(PI2D_DIR)/bin/Pi2D-config --ldflags`
LIBS=`$(PI2D_DIR)/bin/Pi2D-config --libs`

.SUFFIXES: .cpp .o

Test : Test.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *.pyc Test out*.png __pycache__
