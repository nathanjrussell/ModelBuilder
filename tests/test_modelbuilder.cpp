#include <ModelBuilder/ModelBuilder.hpp>
#include <gtest/gtest.h>

#include <filesystem>

TEST(ModelBuilder, BuildTree_Smoke) {
  modelbuilder::ModelBuilder mb;

  const std::filesystem::path dataDir = std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "_deps" /
                                        "contingencytable-src" / "examples" / "datasets" / "titanic_output";

  mb.loadDataDir(dataDir.string());
  ASSERT_GT(mb.rowCount(), 0u);
  ASSERT_GT(mb.columnCount(), 0u);

  // Choose a small target (the dataset has multiple columns; we just want an index within bounds).
  mb.setTargetColumn(0);

  // Constrain analysis columns to a couple candidates if they exist.
  // (This also implicitly tests enabledColumns wiring.)
  if (mb.columnCount() >= 3) {
    mb.setAnalysisColumns({1, 2});
  }

  auto tree = mb.buildTree(/*maxDepth=*/2);
  EXPECT_GT(tree.elementCount(), 0u);
  ASSERT_NE(tree.root(), nullptr);
}

