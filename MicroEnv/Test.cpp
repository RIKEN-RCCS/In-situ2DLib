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

  if ( ! me->execute(std::string("X")) ) {
    printf("MicroEnv::execute failed\n");
    return 3;
  }

  me->finalize();
  delete me;
  return 0;
}
