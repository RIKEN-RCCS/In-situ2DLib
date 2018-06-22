#include "MicroEnv.h"


int main(int argc, char** argv) {
  MicroEnv* me = MicroEnv::GetInstance();
  if ( ! me ) {
    printf("MicroEnv::GetInstance failed\n");
    return 1;
  }

  if ( ! me->initialize() ) {
    printf("MicroEnv::initialize failed\n");
    return 2;
  }

  MicroEnv::DataInfo di;
  di.name = "uvw";
  di.dtype = NPY_DOUBLE;
  di.nd = 2;
  di.dims[0] = 20; di.dims[1] = 10;
  di.p = new double[di.dims[0] * di.dims[1]];
  for ( int j = 0; j < di.dims[0]; j++ )
    for ( int i = 0; i < di.dims[1]; i++ )
      ((double*)di.p)[di.dims[1]*j + i] = (di.dims[1]*j + i)*0.1;
  me->registDmap(di);

  MicroEnv::DataInfo di2;
  di2.name = "idx";
  di2.dtype = NPY_INT;
  di2.nd = 2;
  di2.dims[0] = 20; di2.dims[1] = 10;
  di2.p = new int[di2.dims[0] * di2.dims[1]];
  for ( int j = 0; j < di2.dims[0]; j++ )
    for ( int i = 0; i < di2.dims[1]; i++ )
      ((int*)di2.p)[di2.dims[1]*j + i] = (di2.dims[1]*j + i);
  me->registDmap(di2);

  if ( ! me->execute(std::string("X")) ) {
    printf("MicroEnv::execute failed\n");
    return 3;
  }

  for ( int j = 0; j < di.dims[0]; j++ ) {
    for ( int i = 0; i < di.dims[1]; i++ )
      printf("%.1f ", ((double*)di.p)[di.dims[1]*j + i]);
    printf("\n");
  }

  for ( int j = 0; j < di2.dims[0]; j++ ) {
    for ( int i = 0; i < di2.dims[1]; i++ )
      printf("%d ", ((int*)di2.p)[di2.dims[1]*j + i]);
    printf("\n");
  }
  
  me->finalize();
  delete me;
  return 0;
}
