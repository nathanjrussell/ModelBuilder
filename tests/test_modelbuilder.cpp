#include <ModelBuilder/ModelBuilder.hpp>
#include <gtest/gtest.h>

TEST(ModelBuilder, Add) {
  modelbuilder::ModelBuilder mb;
  EXPECT_EQ(mb.add(2, 3), 5);
}

