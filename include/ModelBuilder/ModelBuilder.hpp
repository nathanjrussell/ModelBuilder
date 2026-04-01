#pragma once

#include <LeftTree/LeftTree.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Forward decl not used anymore; implementation includes the full header.

namespace modelbuilder {

class ModelBuilder {
public:
  struct NodeData {
    std::size_t splitColumnIndex = 0;
    std::vector<std::uint32_t> leftPartitionValues;
    std::vector<std::uint32_t> rightPartitionValues;

    // Number of currently-active rows whose split-column value falls into each partition.
    // Partition "one" corresponds to leftPartitionValues; partition "two" corresponds to rightPartitionValues.
    std::uint64_t leftPartitionCount = 0;
    std::uint64_t rightPartitionCount = 0;

    // Debug: number of currently-enabled rows at the moment this node was created.
    // (Excludes any permanently-disabled header row.)
    std::uint64_t enabledRowCountAtCreation = 0;
  };

  struct LeafData {
    std::uint32_t placeholder = 0;
  };

  using Tree = lefttree::LeftTree<NodeData, LeafData>;

  ModelBuilder();
  ~ModelBuilder();

  ModelBuilder(const ModelBuilder&) = delete;
  ModelBuilder& operator=(const ModelBuilder&) = delete;
  ModelBuilder(ModelBuilder&&) noexcept;
  ModelBuilder& operator=(ModelBuilder&&) noexcept;

  void loadDataDir(const std::string& path);
  void setTargetColumn(std::size_t columnIndex);

  // Significance thresholds used by FeatureSelector.
  // Defaults are the dependency defaults (currently 0.05).
  // If applyBonferroni is true, FeatureSelector will adjust alpha based on the number of tests.
  void setColumnAlpha(double alpha, bool applyBonferroni = true);
  void setPartitionAlpha(double alpha, bool applyBonferroni = false);

  // Columns eligible as split candidates (bitmask passed to FeatureSelector::enabledColumns)
  void setAnalysisColumns(const std::vector<std::size_t>& columnIndices);

  // Build a tree until LeftTree reports complete().
  // - maxDepth = 0 means unlimited.
  [[nodiscard]] Tree buildTree(std::size_t maxDepth = 0);

  // Debug/introspection helpers
  [[nodiscard]] std::uint64_t rowCount() const;
  [[nodiscard]] std::uint64_t columnCount() const;

  // Writes a Graphviz .dot file representing the provided tree.
  // Each element is labeled with its LeftTree element id and any stored payload.
  void createGraphviz(const Tree& tree, const std::string& outputDotPath) const;

private:
  struct Impl;
  Impl* impl_;
};

} // namespace modelbuilder

