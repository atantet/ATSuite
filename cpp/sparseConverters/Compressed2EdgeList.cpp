#include <cstdlib>
#include <cstdio>
#include "atio.hpp"

int main(int argc, char *argv[])
{
  FILE *src, *dst;
  
  if ((src = fopen(argv[1], "r")) == NULL){
    std::cerr << "Can't open " << argv[1] << " for reading!" << std::endl;
    return -1;
  }
  if ((dst = fopen(argv[2], "w")) == NULL){
    std::cerr << "Can't open " << argv[2] << " for writting!" << std::endl;
    return -1;
  }

  Compressed2EdgeList(src, dst);

  return 0;
}
