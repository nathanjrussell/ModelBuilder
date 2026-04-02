#include <ModelBuilder/ModelBuilder.hpp>
#include <ModelBuilder/MultiModelBuilder.hpp>
#include <ModelBuilder/TreeBuilder.hpp>
#include <DataTable/DataTable.h>
#include <gtest/gtest.h>

#include <LeftTree/LeftTreeSerialization.hpp>

#include <filesystem>
#include <map>
#include <unordered_set>

namespace {
std::filesystem::path ensureParsedPtsdDataset() {
  const std::filesystem::path outDir =
      std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "datasets" / "ptsd_output";

  const auto metaDir = outDir / "meta_data";
  const auto mappedDir = outDir / "mapped_data";
  if (!std::filesystem::exists(metaDir) || !std::filesystem::exists(mappedDir)) {
    throw std::runtime_error(std::string{"Missing parsed PTSD dataset at: "} + outDir.string() +
                             " (run the examples with --reparse to generate it)");
  }

  return outDir;
}
} // namespace

TEST(ModelBuilder, BuildTree_Smoke) {
  modelbuilder::ModelBuilder mb;

  const std::filesystem::path dataDir = ensureParsedPtsdDataset();

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
        EXPECT_GE(nd->leftPartitionCount(), 0u);
        EXPECT_GE(nd->rightPartitionCount(), 0u);
      }
    }
  }
}

namespace {
std::uint64_t nodeTotalCount(const modelbuilder::NodeData& nd) {
  return nd.leftPartitionCount() + nd.rightPartitionCount();
}
} // namespace

TEST(ModelBuilder, BuildTree_ChildTotalsMatchParentBranchCount) {
  modelbuilder::ModelBuilder mb;

  // Use deterministic built-in dataset from the dependency.
  const std::filesystem::path dataDir = ensureParsedPtsdDataset();

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
        const auto expected = pnd.leftPartitionCount();
        const auto actual = nodeTotalCount(*leftChild->data);
        EXPECT_EQ(actual, expected) << "parent=" << parent->id << " leftChild=" << leftChild->id;
      }
    }

    // Right branch
    if (parent->right && !parent->right->isLeaf()) {
      const auto* rightChild = tree.getNode(parent->right->id);
      ASSERT_NE(rightChild, nullptr);
      if (rightChild->data.has_value()) {
        const auto expected = pnd.rightPartitionCount();
        const auto actual = nodeTotalCount(*rightChild->data);
        EXPECT_EQ(actual, expected) << "parent=" << parent->id << " rightChild=" << rightChild->id;
      }
    }
  }
}

TEST(ModelBuilder, TreeSerialization_RoundTrip) {
  modelbuilder::ModelBuilder mb;

  const std::filesystem::path dataDir = ensureParsedPtsdDataset();

  mb.loadDataDir(dataDir.string());
  mb.setTargetColumn(1);
  mb.setAnalysisColumns({2});

  auto tree = mb.buildTree(/*maxDepth=*/2);
  ASSERT_NE(tree.root(), nullptr);

  using Ser = lefttree::LeftTreeSerialization<modelbuilder::NodeData, modelbuilder::LeafData>;
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  Ser::serializeToStream(tree, ss);

  ss.seekg(0);
  auto tree2 = Ser::deserializeFromStream(ss);
  ASSERT_NE(tree2.root(), nullptr);
  EXPECT_EQ(tree2.elementCount(), tree.elementCount());

  // Verify leaf payload parity (including targetCounts) across the whole tree.
  for (std::size_t i = 0; i < tree.elementCount(); ++i) {
    const auto d0i = tree.getElementData(i);
    const auto d1i = tree2.getElementData(i);
    ASSERT_TRUE(d0i.has_value());
    ASSERT_TRUE(d1i.has_value());

    if (std::holds_alternative<modelbuilder::ModelBuilder::Tree::LeafDataResult>(*d0i)) {
      ASSERT_TRUE(std::holds_alternative<modelbuilder::ModelBuilder::Tree::LeafDataResult>(*d1i));
      const auto& [t0l, l0] = std::get<modelbuilder::ModelBuilder::Tree::LeafDataResult>(*d0i);
      const auto& [t1l, l1] = std::get<modelbuilder::ModelBuilder::Tree::LeafDataResult>(*d1i);
      (void)t0l;
      (void)t1l;
      ASSERT_TRUE(l0.has_value());
      ASSERT_TRUE(l1.has_value());
      EXPECT_EQ(l0->reason(), l1->reason());
      EXPECT_EQ(l0->splitColumnIndexOrMinusOne(), l1->splitColumnIndexOrMinusOne());
      EXPECT_EQ(l0->targetCounts(), l1->targetCounts());
    }
  }

  // Spot-check root payload parity when both are nodes with payload.
  auto d0 = tree.getElementData(0);
  auto d1 = tree2.getElementData(0);
  ASSERT_TRUE(d0.has_value());
  ASSERT_TRUE(d1.has_value());
  ASSERT_TRUE(std::holds_alternative<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*d0));
  ASSERT_TRUE(std::holds_alternative<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*d1));

  const auto& [t0, n0] = std::get<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*d0);
  const auto& [t1, n1] = std::get<modelbuilder::ModelBuilder::Tree::NodeDataResult>(*d1);
  (void)t0;
  (void)t1;
  ASSERT_TRUE(n0.has_value());
  ASSERT_TRUE(n1.has_value());
  EXPECT_EQ(n0->splitColumnIndex(), n1->splitColumnIndex());
  EXPECT_EQ(n0->leftPartitionCount(), n1->leftPartitionCount());
  EXPECT_EQ(n0->rightPartitionCount(), n1->rightPartitionCount());
  EXPECT_EQ(n0->leftPartitionValues(), n1->leftPartitionValues());
  EXPECT_EQ(n0->rightPartitionValues(), n1->rightPartitionValues());
}

TEST(TreeBuilder, ArtifactSerialization_RoundTrip) {
  // Use deterministic built-in dataset from a dependency.
  const std::filesystem::path dataDir = ensureParsedPtsdDataset();

  const std::size_t targetColumn = 1;
  const std::size_t maxDepth = 2;
  const double columnAlpha = 0.05;
  const bool columnBonferroni = true;
  const double partitionAlpha = 0.01;
  const bool partitionBonferroni = false;

  // Build using a constrained analysis column set (build-time only).
  auto a1 = modelbuilder::TreeBuilder::buildFromDataDir(dataDir.string(),
                                                       targetColumn,
                                                       maxDepth,
                                                       columnAlpha,
                                                       columnBonferroni,
                                                       partitionAlpha,
                                                       partitionBonferroni,
                                                       /*analysisColumns=*/{2});
  ASSERT_NE(a1.tree().root(), nullptr);

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  a1.serialize(ss);

  ss.seekg(0);
  auto a2 = modelbuilder::TreeBuilder::deserialize(ss);
  ASSERT_NE(a2.tree().root(), nullptr);

  // Ensure the pinned params round-trip.
  EXPECT_EQ(a2.targetColumn(), a1.targetColumn());
  EXPECT_EQ(a2.maxDepth(), a1.maxDepth());
  EXPECT_DOUBLE_EQ(a2.columnAlpha(), a1.columnAlpha());
  EXPECT_EQ(a2.columnAlphaApplyBonferroni(), a1.columnAlphaApplyBonferroni());
  EXPECT_DOUBLE_EQ(a2.partitionAlpha(), a1.partitionAlpha());
  EXPECT_EQ(a2.partitionAlphaApplyBonferroni(), a1.partitionAlphaApplyBonferroni());

  // Tree blob round-trip sanity.
  EXPECT_EQ(a2.tree().elementCount(), a1.tree().elementCount());

  // Spot-check root payload parity (node or leaf).
  auto d0 = a1.tree().getElementData(0);
  auto d1 = a2.tree().getElementData(0);
  ASSERT_TRUE(d0.has_value());
  ASSERT_TRUE(d1.has_value());

  if (std::holds_alternative<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*d0)) {
    ASSERT_TRUE(std::holds_alternative<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*d1));
    const auto& [t0, n0] = std::get<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*d0);
    const auto& [t1, n1] = std::get<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*d1);
    (void)t0;
    (void)t1;
    ASSERT_TRUE(n0.has_value());
    ASSERT_TRUE(n1.has_value());
    EXPECT_EQ(n0->splitColumnIndex(), n1->splitColumnIndex());
    EXPECT_EQ(n0->leftPartitionCount(), n1->leftPartitionCount());
    EXPECT_EQ(n0->rightPartitionCount(), n1->rightPartitionCount());
    EXPECT_EQ(n0->leftPartitionValues(), n1->leftPartitionValues());
    EXPECT_EQ(n0->rightPartitionValues(), n1->rightPartitionValues());
  } else {
    ASSERT_TRUE(std::holds_alternative<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*d0));
    ASSERT_TRUE(std::holds_alternative<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*d1));
    const auto& [t0, l0] = std::get<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*d0);
    const auto& [t1, l1] = std::get<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*d1);
    (void)t0;
    (void)t1;
    ASSERT_TRUE(l0.has_value());
    ASSERT_TRUE(l1.has_value());
    EXPECT_EQ(l0->reason(), l1->reason());
    EXPECT_EQ(l0->splitColumnIndexOrMinusOne(), l1->splitColumnIndexOrMinusOne());
    EXPECT_EQ(l0->targetCounts(), l1->targetCounts());
  }
}

TEST(MultiModelBuilder, Predict_NormalizesLeafCounts) {
  const std::filesystem::path dataDir = ensureParsedPtsdDataset();
  const std::filesystem::path outDir = std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "tests" /
                                       "predict_model";

  // Create a small multi-model with one target.
  auto mb = modelbuilder::MultiModelBuilder::buildAndWrite(dataDir.string(),
                                                           outDir.string(),
                                                           /*targetColumns=*/{1},
                                                           /*maxDepth=*/2,
                                                           /*columnAlpha=*/0.05,
                                                           /*columnAlphaApplyBonferroni=*/true,
                                                           /*partitionAlpha=*/0.05,
                                                           /*partitionAlphaApplyBonferroni=*/false,
                                                           /*analysisColumns=*/{2},
                                                           /*threadCount=*/1);
  mb.loadMap();

  // Use all-missing sample so we can traverse conditionally regardless of split columns.
  const std::vector<std::uint64_t> sample(/*cols=*/16, 0);
  const auto dist = mb.predict(sample, /*targetColumn=*/1, /*applyConditional=*/true);
  ASSERT_FALSE(dist.empty());
  double sum = 0.0;
  for (const auto& kv : dist) {
    EXPECT_GE(kv.second, 0.0);
    sum += kv.second;
  }
  EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(MultiModelBuilder, Predict_MissingValueRequiresConditional) {
  const std::filesystem::path dataDir = ensureParsedPtsdDataset();
  const std::filesystem::path outDir = std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "tests" /
                                       "predict_model_2";

  auto mb = modelbuilder::MultiModelBuilder::buildAndWrite(dataDir.string(),
                                                           outDir.string(),
                                                           /*targetColumns=*/{1},
                                                           /*maxDepth=*/2,
                                                           /*columnAlpha=*/0.05,
                                                           /*columnAlphaApplyBonferroni=*/true,
                                                           /*partitionAlpha=*/0.05,
                                                           /*partitionAlphaApplyBonferroni=*/false,
                                                           /*analysisColumns=*/{2},
                                                           /*threadCount=*/1);
  mb.loadMap();

  const std::vector<std::uint64_t> sample(/*cols=*/16, 0);
  EXPECT_THROW((void)mb.predict(sample, /*targetColumn=*/1, /*applyConditional=*/false), std::exception);
}

TEST(MultiModelBuilder, Predict_UnknownValueThrows) {
  const std::filesystem::path dataDir = ensureParsedPtsdDataset();
  const std::filesystem::path outDir = std::filesystem::path{MODELBUILDER_SOURCE_DIR} / "build" / "tests" /
                                       "predict_model_3";

  auto mb = modelbuilder::MultiModelBuilder::buildAndWrite(dataDir.string(),
                                                           outDir.string(),
                                                           /*targetColumns=*/{1},
                                                           /*maxDepth=*/2,
                                                           /*columnAlpha=*/0.05,
                                                           /*columnAlphaApplyBonferroni=*/true,
                                                           /*partitionAlpha=*/0.05,
                                                           /*partitionAlphaApplyBonferroni=*/false,
                                                           /*analysisColumns=*/{2},
                                                           /*threadCount=*/1);
  mb.loadMap();

  // Construct a sample that sets split columns to a value very likely not in any partition.
  // We discover the root split column from the tree payload.
  auto art = mb.getTree(1);
  auto rootOpt = art.tree().getElementData(0);
  ASSERT_TRUE(rootOpt.has_value());
  ASSERT_TRUE(std::holds_alternative<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*rootOpt));
  const auto& [t, ndOpt] = std::get<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*rootOpt);
  (void)t;
  ASSERT_TRUE(ndOpt.has_value());
  const auto split = static_cast<std::size_t>(ndOpt->splitColumnIndex());

  std::vector<std::uint64_t> sample(/*cols=*/16, 0);
  sample[split] = 9999999; // unknown
  EXPECT_THROW((void)mb.predict(sample, /*targetColumn=*/1, /*applyConditional=*/true), std::exception);
}

