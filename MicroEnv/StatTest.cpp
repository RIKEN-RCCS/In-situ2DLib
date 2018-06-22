#include <sys/stat.h>
#include <time.h>
#include <cstdio>
#include <unistd.h>

int main(int argc, char** argv) {
  const char* fpath = "/tmp/AAA";
  struct stat sbuf;
  time_t now = time(NULL);

  printf("sleep 5 sec, touch %s if you want.\n", fpath);
  sleep(5);
  
  if ( stat(fpath, &sbuf) ) {
    fprintf(stderr, "stat failed: %s\n", fpath);
    return 1;
  }

  printf("mtime = %s\n", ctime(&sbuf.st_mtime));
  if ( now > sbuf.st_mtime )
    printf("%s not modified\n", fpath);
  else
    printf("%s modified\n", fpath);
  return 0;
}
