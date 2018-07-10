#include "Pi2D.h"
#include <cstdio>

int main(int argc, char** argv) {
  Pi2D p;

  if ( ! p.ImportAttrib("Y.json") ) {
    fprintf(stderr, "import Y.json failed.\n");
    return 1;
  }
  return 0;
}
