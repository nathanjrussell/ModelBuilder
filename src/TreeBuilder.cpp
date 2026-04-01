#include <ModelBuilder/TreeBuilder.hpp>

#include <ContingencyTable/FeatureSelector.h>

#include <limits>
#include <stdexcept>

namespace modelbuilder {

namespace {
std::vector<std::uint32_t> makeBitmaskFromIndices(std::size_t numBits, const std::vector<std::size_t>& indices) {
  const std::size_t numWords = (numBits + 31) / 32;
  std::vector<std::uint32_t> bm(numWords, 0);
  for (std::size_t idx : indices) {
    if (idx >= numBits) {
      throw std::out_of_range("TreeBuilder: column index out of range");
    }
    bm[idx / 32] |= (1u << (idx % 32));
  }
  return bm;
}
} // namespace

TreeBuilder TreeBuilder::buildFromDataDir(const std::string& parsedDir,
                                         std::size_t targetColumn,
                                         std::size_t maxDepth,
                                         double columnAlpha,
                                         bool columnAlphaApplyBonferroni,
                                         double partitionAlpha,
                                         bool partitionAlphaApplyBonferroni,
                                         const std::vector<std::size_t>& analysisColumns) {
  ContingencyTableLib::FeatureSelector selector;
  selector.load(parsedDir);

  const auto rows = selector.getRowCount();
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("TreeBuilder: row count too large for internal buffers");
  }
  std::vector<std::uint32_t> rowIndices(static_cast<std::size_t>(rows), 0);
  if (!rowIndices.empty()) {
    // DataTable underlying ContingencyTable uses row 0 as header row.
    rowIndices[0] = std::numeric_limits<std::uint32_t>::max();
  }

  if (targetColumn >= selector.getColumnCount()) {
    throw std::out_of_range("TreeBuilder: target column out of range");
  }
  selector.setTargetColumn(targetColumn);

  selector.setColumnAlpha(columnAlpha, columnAlphaApplyBonferroni);
  selector.setPartitionAlpha(partitionAlpha, partitionAlphaApplyBonferroni);

  // Build-time only: configure split candidate columns.
  // FeatureSelector requires enabledColumns() to be called before findSignificantColumn().
  //
  // If the caller provides an analysis column list, use it verbatim.
  // Otherwise, default to enabling all columns EXCEPT the target column.
  const auto colsU64 = selector.getColumnCount();
  if (colsU64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("TreeBuilder: column count too large for internal buffers");
  }
  const auto cols = static_cast<std::size_t>(colsU64);

  std::vector<std::uint32_t> enabledColumnsBitmask;
  if (!analysisColumns.empty()) {
    enabledColumnsBitmask = makeBitmaskFromIndices(cols, analysisColumns);
  } else {
    // Enable all columns.
    enabledColumnsBitmask.assign((cols + 31) / 32, 0xFFFFFFFFu);
    // Clear any padding bits above cols.
    const std::size_t extraBits = enabledColumnsBitmask.size() * 32 - cols;
    if (extraBits != 0) {
      enabledColumnsBitmask.back() >>= extraBits;
    }
    // Always exclude the target column from split candidates.
    if (targetColumn < cols) {
      enabledColumnsBitmask[targetColumn / 32] &= ~(1u << (targetColumn % 32));
    }
  }
  selector.enabledColumns(enabledColumnsBitmask.data(), cols);

  Tree tree;

  auto countEnabledRowsAfterApplyingPartition = [&](std::uint32_t threshold,
                                                    std::size_t splitColumnIndex,
                                                    const std::vector<std::uint32_t>& allowedValues) -> std::uint64_t {
    // Take a copy so we can apply an additional filter without mutating the live marker state.
    auto tmp = rowIndices;
    selector.enabledRows(tmp, threshold, allowedValues, splitColumnIndex);
    std::uint64_t c = 0;
    for (std::size_t i = 1; i < tmp.size(); ++i) {
      if (tmp[i] == 0) ++c;
    }
    return c;
  };

  auto rebuildEnabledRowsForCurrentPosition = [&]() {
    // Recompute selector row state for the current insertion point by applying
    // the sequence of splits from the root down to the *parent* of the next element.
    // This matches ModelBuilder::buildTree().
    const auto* cur = tree.getCurrentNode();
    if (!cur) return;

    // Clear markers >= current node id.
    const auto curIdU32 = static_cast<std::uint32_t>(cur->id);
    for (auto& m : rowIndices) {
      if (m >= curIdU32) m = 0;
    }
    if (!rowIndices.empty()) {
      rowIndices[0] = std::numeric_limits<std::uint32_t>::max();
    }

    // Collect path from root to current.
    std::vector<const Tree::Node*> path;
    for (const Tree::Node* n = cur; n != nullptr; ) {
      path.push_back(n);
      const Tree::Node* parent = nullptr;
      for (Tree::Id pid = n->id - 1; pid >= 0; --pid) {
        const auto* cand = tree.getNode(pid);
        if (!cand) continue;
        if ((cand->left && cand->left.get() == n) || (cand->right && cand->right.get() == n)) {
          parent = cand;
          break;
        }
      }
      n = parent;
    }
    std::reverse(path.begin(), path.end());

    if (path.size() <= 1) return;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
      const auto* n = path[i];
      if (!n->data.has_value()) continue;
      const auto& nd = *n->data;
      const auto* child = path[i + 1];
      const bool tookLeft = (n->left && n->left.get() == child);
      const auto& allowed = tookLeft ? nd.leftPartitionValues() : nd.rightPartitionValues();
      selector.enabledRows(rowIndices,
                          static_cast<std::uint32_t>(n->id),
                          allowed,
                          static_cast<std::size_t>(nd.splitColumnIndex()));
    }

    if (auto* insertionParent = tree.getCurrentNode(); insertionParent && insertionParent->data.has_value()) {
      const auto& nd = *insertionParent->data;
      const bool insertingLeft = (insertionParent->left == nullptr);
      const auto& allowed = insertingLeft ? nd.leftPartitionValues() : nd.rightPartitionValues();
      selector.enabledRows(rowIndices,
                          static_cast<std::uint32_t>(insertionParent->id),
                          allowed,
                          static_cast<std::size_t>(nd.splitColumnIndex()));
    }
  };

  // Seed a root element.
  selector.findSignificantColumn();
  if (!selector.significantColumnFound() || !selector.significantPartitionFound()) {
    if (!selector.significantColumnFound()) {
      tree.createLeaf(LeafData::noSignificantColumn(selector.getTargetCounts()));
    } else {
      tree.createLeaf(LeafData::noSignificantPartition(selector.getSignificantColumnIndex(),
                                                       selector.getTargetCounts()));
    }
  } else {
    const auto nodeIdForCounts = static_cast<std::uint32_t>(tree.currentNodeID() < 0 ? 0 : tree.currentNodeID());
    const auto splitColumnIndex = static_cast<std::uint64_t>(selector.getSignificantColumnIndex());
    auto leftValues = selector.getFirstPartition();
    auto rightValues = selector.getSecondPartition();
    const auto countThreshold = nodeIdForCounts + 1;
    const auto leftCount =
      countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), leftValues);
    const auto rightCount =
      countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), rightValues);

    tree.createNode(NodeData{splitColumnIndex, std::move(leftValues), std::move(rightValues), leftCount, rightCount});
  }

  // Depth tracking aligned to insertion path.
  std::vector<Tree::Id> openNodeStack;
  std::vector<std::size_t> depthStack;
  auto currentDepth = [&]() -> std::size_t {
    if (depthStack.empty()) return 0;
    return depthStack.back();
  };

  while (!tree.complete()) {
    if (maxDepth != 0 && currentDepth() >= maxDepth) {
      // Depth-limited leaf; treat as "no significant column" for now.
      // We still call findSignificantColumn() elsewhere to populate target counts; but at a
      // depth-limit leaf we want the target distribution of the rows that reached the node.
      // Rebuild enabled rows, run feature selection just to compute target counts.
      rebuildEnabledRowsForCurrentPosition();
      selector.findSignificantColumn();
      tree.createLeaf(LeafData::noSignificantColumn(selector.getTargetCounts()));
    } else {
      rebuildEnabledRowsForCurrentPosition();
      selector.findSignificantColumn();

      if (!selector.significantColumnFound() || !selector.significantPartitionFound()) {
        if (!selector.significantColumnFound()) {
          tree.createLeaf(LeafData::noSignificantColumn(selector.getTargetCounts()));
        } else {
          tree.createLeaf(LeafData::noSignificantPartition(selector.getSignificantColumnIndex(),
                                                          selector.getTargetCounts()));
        }
      } else {
        const auto nodeIdForCounts = static_cast<std::uint32_t>(tree.currentNodeID() < 0 ? 0 : tree.currentNodeID());
        const auto splitColumnIndex = static_cast<std::uint64_t>(selector.getSignificantColumnIndex());
        auto leftValues = selector.getFirstPartition();
        auto rightValues = selector.getSecondPartition();

        const auto countThreshold = nodeIdForCounts + 1;
        const auto leftCount =
          countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), leftValues);
        const auto rightCount =
          countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), rightValues);

        tree.createNode(NodeData{splitColumnIndex,
                                std::move(leftValues),
                                std::move(rightValues),
                                leftCount,
                                rightCount});
      }
    }

    auto* cur = tree.getCurrentNode();
    if (!cur) break;

    if (openNodeStack.empty() || openNodeStack.back() != cur->id) {
      if (!cur->left && !cur->right) {
        const std::size_t d = depthStack.empty() ? 1 : (depthStack.back() + 1);
        openNodeStack.push_back(cur->id);
        depthStack.push_back(d);
      } else {
        while (!openNodeStack.empty() && openNodeStack.back() != cur->id) {
          openNodeStack.pop_back();
          depthStack.pop_back();
        }
      }
    }
  }

  return TreeBuilder{targetColumn,
                     maxDepth,
                     columnAlpha,
                     columnAlphaApplyBonferroni,
                     partitionAlpha,
                     partitionAlphaApplyBonferroni,
                     std::move(tree)};
}

} // namespace modelbuilder


