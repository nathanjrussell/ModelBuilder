#include <ModelBuilder/ModelBuilder.hpp>

#include <iostream>

int main() {
  modelbuilder::ModelBuilder mb;
  std::cout << "2 + 3 = " << mb.add(2, 3) << "\n";
  return 0;
}

