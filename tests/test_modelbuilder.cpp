#include <ModelBuilder/ModelBuilder.hpp>
#include <DataTable/DataTable.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <unordered_set>

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

  // If the root element is a node and has payload, verify the new count fields are accessible.
  if (auto dataOpt = tree.getElementData(0); dataOpt.has_value()) {
    if (std::holds_alternative<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*dataOpt)) {
      const auto& [elemType, nd] = std::get<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*dataOpt);
      (void)elemType;
      if (nd.has_value()) {
        EXPECT_GE(nd->leftPartitionCount, 0u);
        EXPECT_GE(nd->rightPartitionCount, 0u);
      }
    }
  }
}

namespace {
std::uint64_t nodeTotalCount(const modelbuilder::ModelBuilder::NodeData& nd) {
  return nd.leftPartitionCount + nd.rightPartitionCount;
}
} // namespace

TEST(ModelBuilder, BuildTree_ChildTotalsMatchParentBranchCount) {
  modelbuilder::ModelBuilder mb;

  // Use deterministic built-in dataset from the dependency.
  const std::filesystem::path dataDir = std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "_deps" /
                                        "contingencytable-src" / "examples" / "datasets" / "titanic_output";

  mb.loadDataDir(dataDir.string());
  ASSERT_GT(mb.rowCount(), 0u);
  ASSERT_GT(mb.columnCount(), 2u);

  mb.setTargetColumn(1);
  // Enable a handful of analysis columns to ensure we get at least a couple internal nodes.
  mb.setAnalysisColumns({2});

  auto tree = mb.buildTree(/*maxDepth=*/3);
  ASSERT_NE(tree.root(), nullptr);

  // Walk all internal nodes and verify that if a child is also an internal node then:
  //   child.leftCount + child.rightCount == parent.<branch>Count
  // This is the invariant you described.
  //
  // IMPORTANT: Counts in ModelBuilder are defined as the number of enabled rows that reach each
  // branch (after applying the parent's partition filter), consistent with the row-marker scheme.
  for (std::size_t i = 0; i < tree.elementCount(); ++i) {
    const auto id = static_cast<modelbuilder::ModelBuilder::Tree::Id>(i);
    const auto* parent = tree.getNode(id);
    if (!parent || !parent->data.has_value()) continue;

    const auto& pnd = *parent->data;

    // Left branch
    if (parent->left && !parent->left->isLeaf()) {
      const auto* leftChild = tree.getNode(parent->left->id);
      ASSERT_NE(leftChild, nullptr);
      if (leftChild->data.has_value()) {
        const auto expected = pnd.leftPartitionCount;
        const auto actual = nodeTotalCount(*leftChild->data);
        EXPECT_EQ(actual, expected) << "parent=" << parent->id << " leftChild=" << leftChild->id;

        // Sanity check: the child should have been created under that branch's enabled set.
        EXPECT_EQ(leftChild->data->enabledRowCountAtCreation, expected)
            << "parent=" << parent->id << " leftChild=" << leftChild->id << " (enabled at creation)";
      }
    }

    // Right branch
    if (parent->right && !parent->right->isLeaf()) {
      const auto* rightChild = tree.getNode(parent->right->id);
      ASSERT_NE(rightChild, nullptr);
      if (rightChild->data.has_value()) {
        const auto expected = pnd.rightPartitionCount;
        const auto actual = nodeTotalCount(*rightChild->data);
        EXPECT_EQ(actual, expected) << "parent=" << parent->id << " rightChild=" << rightChild->id;

        EXPECT_EQ(rightChild->data->enabledRowCountAtCreation, expected)
            << "parent=" << parent->id << " rightChild=" << rightChild->id << " (enabled at creation)";
      }
    }
  }
}

