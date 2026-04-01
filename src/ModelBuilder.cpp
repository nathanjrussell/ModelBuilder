#include <ModelBuilder/ModelBuilder.hpp>

#include <ContingencyTable/FeatureSelector.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace modelbuilder {

struct ModelBuilder::Impl {
  ContingencyTableLib::FeatureSelector selector;

  std::string dataPath;
  bool loaded = false;

  std::size_t targetColumn = 0;

  // Row marker vector used by FeatureSelector::enabledRows(vector<uint32_t>&,...)
  std::vector<std::uint32_t> rowIndices;

  // Enabled columns bitmask for FeatureSelector::enabledColumns
  std::vector<std::uint32_t> enabledColumnsBitmask;
  bool hasAnalysisColumns = false;

  void ensureLoaded() const {
    if (!loaded) throw std::logic_error("ModelBuilder: call loadDataDir() before building");
  }

  static std::vector<std::uint32_t> makeBitmaskFromIndices(std::size_t numBits,
                                                          const std::vector<std::size_t>& indices) {
    const std::size_t numWords = (numBits + 31) / 32;
    std::vector<std::uint32_t> bm(numWords, 0);
    for (std::size_t idx : indices) {
      if (idx >= numBits) {
        throw std::out_of_range("ModelBuilder: column index out of range");
      }
      bm[idx / 32] |= (1u << (idx % 32));
    }
    return bm;
  }
};

ModelBuilder::ModelBuilder() : impl_(new Impl()) {}
ModelBuilder::~ModelBuilder() { delete impl_; }
ModelBuilder::ModelBuilder(ModelBuilder&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }
ModelBuilder& ModelBuilder::operator=(ModelBuilder&& other) noexcept {
  if (this == &other) return *this;
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

void ModelBuilder::loadDataDir(const std::string& path) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->dataPath = path;
  impl_->selector.load(path);
  impl_->loaded = true;

  const auto rows = impl_->selector.getRowCount();
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("ModelBuilder: row count too large for internal buffers");
  }
  impl_->rowIndices.assign(static_cast<std::size_t>(rows), 0);
  if (!impl_->rowIndices.empty()) {
    // DataTable underlying ContingencyTable uses row 0 as header row.
    // Keep it permanently disabled in the marker scheme.
    impl_->rowIndices[0] = std::numeric_limits<std::uint32_t>::max();
  }
}

void ModelBuilder::setTargetColumn(std::size_t columnIndex) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->targetColumn = columnIndex;
  if (impl_->loaded) {
    if (columnIndex >= impl_->selector.getColumnCount()) {
      throw std::out_of_range("ModelBuilder: target column out of range");
    }
    impl_->selector.setTargetColumn(columnIndex);
  }
}

void ModelBuilder::setColumnAlpha(double alpha, bool applyBonferroni) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();
  impl_->selector.setColumnAlpha(alpha, applyBonferroni);
}

void ModelBuilder::setPartitionAlpha(double alpha, bool applyBonferroni) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();
  impl_->selector.setPartitionAlpha(alpha, applyBonferroni);
}

void ModelBuilder::setAnalysisColumns(const std::vector<std::size_t>& columnIndices) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();

  const auto cols = impl_->selector.getColumnCount();
  if (cols > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("ModelBuilder: column count too large for internal buffers");
  }
  impl_->enabledColumnsBitmask = Impl::makeBitmaskFromIndices(static_cast<std::size_t>(cols), columnIndices);
  impl_->hasAnalysisColumns = true;

  impl_->selector.enabledColumns(impl_->enabledColumnsBitmask.data(), static_cast<std::size_t>(cols));
}

ModelBuilder::Tree ModelBuilder::buildTree(std::size_t maxDepth) {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();

  if (impl_->targetColumn >= impl_->selector.getColumnCount()) {
    throw std::out_of_range("ModelBuilder: target column out of range");
  }
  impl_->selector.setTargetColumn(impl_->targetColumn);

  // If analysis columns not set, default to "all columns" and let FeatureSelector do its own exclusions
  // (typically it excludes target column internally).
  if (impl_->hasAnalysisColumns) {
    const auto cols = impl_->selector.getColumnCount();
    impl_->selector.enabledColumns(impl_->enabledColumnsBitmask.data(), static_cast<std::size_t>(cols));
  }

  Tree tree;

  auto countEnabledRowsAfterApplyingPartition = [&](std::uint32_t threshold,
                                                    std::size_t splitColumnIndex,
                                                    const std::vector<std::uint32_t>& allowedValues) -> std::uint64_t {
    // Take a copy so we can apply an additional filter without mutating the live marker state.
    auto tmp = impl_->rowIndices;
    impl_->selector.enabledRows(tmp, threshold, allowedValues, splitColumnIndex);
    std::uint64_t c = 0;
    for (std::size_t i = 1; i < tmp.size(); ++i) {
      if (tmp[i] == 0) ++c;
    }
    return c;
  };

  auto rebuildEnabledRowsForCurrentPosition = [&]() {
    // Recompute selector row state for the current insertion point by applying
    // the sequence of splits from the root down to the *parent* of the next element.
    // This avoids subtle inconsistencies when LeftTree backtracks from a fully-grown
    // left branch to a right branch.
    const auto* cur = tree.getCurrentNode();
    if (!cur) return;

    // Restore row marker state to what it looked like when we *arrived* at the current node.
    // Contract (as used by FeatureSelector::enabledRows(rowIndices,...)):
    // - When a node with id=N applies a partition filter, any row excluded is marked with value N.
    // - When backtracking to (or re-evaluating under) node N, we must re-enable all rows that were
    //   excluded by node N or any deeper node, i.e. clear markers >= N back to 0.
    const auto curIdU32 = static_cast<std::uint32_t>(cur->id);
    for (auto& m : impl_->rowIndices) {
      if (m >= curIdU32) m = 0;
    }

    // Keep DataTable header row permanently disabled in our marker scheme.
    if (!impl_->rowIndices.empty()) {
      impl_->rowIndices[0] = std::numeric_limits<std::uint32_t>::max();
    }

    // Collect the path nodes from root to current.
    std::vector<const Tree::Node*> path;
    for (const Tree::Node* n = cur; n != nullptr; ) {
      path.push_back(n);

      // Find parent by scanning ids downwards (IDs are sequential and parent is < child).
      const Tree::Node* parent = nullptr;
      for (Tree::Id pid = n->id - 1; pid >= 0; --pid) {
        const auto* cand = tree.getNode(pid);
        if (!cand) continue;
        // Use pointer identity (not just id) to avoid mismatches during construction.
        if ((cand->left && cand->left.get() == n) || (cand->right && cand->right.get() == n)) {
          parent = cand;
          break;
        }
      }
      n = parent;
    }
    std::reverse(path.begin(), path.end());

    // Apply each ancestor split on the way down, but only for ancestors of the current insertion node.
    // The current insertion node's own split has NOT been taken yet (we are about to decide
    // its left vs right child), so applying it here would incorrectly filter out rows.
    if (path.size() <= 1) return;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
      const auto* n = path[i];
      if (!n->data.has_value()) continue;
      const auto& nd = *n->data;

      // Determine which branch we took from this node.
      const auto* child = path[i + 1];
      const bool tookLeft = (n->left && n->left.get() == child);

      // IMPORTANT: By contract, FeatureSelector partition "one" corresponds to the left branch
      // (LeftTree edge label "L") and partition "two" corresponds to the right branch.
      // NodeData stores these as leftPartitionValues/rightPartitionValues.
      const auto& allowed = tookLeft ? nd.leftPartitionValues() : nd.rightPartitionValues();

      // Use threshold == node id.
      // This means: before applying this node's whitelist, re-enable any rows that were previously
      // disabled at this node or deeper (marker >= nodeId).
      impl_->selector.enabledRows(impl_->rowIndices,
                                 static_cast<std::uint32_t>(n->id),
                                 allowed,
                                 static_cast<std::size_t>(nd.splitColumnIndex()));
    }

    // Finally, apply the *current node's* branch filter for the next insertion (left child first,
    // and after backtracking, right child). This is the critical step that makes the enabled rows
    // for the soon-to-be-created child match the parent's left/right partition counts.
    if (auto* insertionParent = tree.getCurrentNode(); insertionParent && insertionParent->data.has_value()) {
      const auto& nd = *insertionParent->data;
      const bool insertingLeft = (insertionParent->left == nullptr);
      const auto& allowed = insertingLeft ? nd.leftPartitionValues() : nd.rightPartitionValues();
      impl_->selector.enabledRows(impl_->rowIndices,
                                 static_cast<std::uint32_t>(insertionParent->id),
                                 allowed,
                                 static_cast<std::size_t>(nd.splitColumnIndex()));
    }
  };

  // LeftTree considers an empty tree "complete" (current node == nullptr). Seed a root element.
  // The root decision is based on feature selection on all currently enabled rows.
  impl_->selector.findSignificantColumn();
  if (!impl_->selector.significantColumnFound() || !impl_->selector.significantPartitionFound()) {
    tree.createLeaf(LeafData{0});
  } else {
    const auto nodeIdForCounts = static_cast<std::uint32_t>(tree.currentNodeID() < 0 ? 0 : tree.currentNodeID());
    const auto splitColumnIndex = static_cast<std::uint64_t>(impl_->selector.getSignificantColumnIndex());
    // Partition "one" is left, partition "two" is right.
    auto leftValues = impl_->selector.getFirstPartition();
    auto rightValues = impl_->selector.getSecondPartition();

    // Counts are defined as the number of rows reaching each branch (child) of this node.
    // Compute using the current enabled set (rows reaching this node) and applying the node's
    // own partition whitelist to a temporary marker vector.
    // When applying a hypothetical partition filter for counting purposes, we must not clear
    // any markers that represent the current node's arrival state. Since enabledRows() clears
    // markers >= threshold, use (nodeId + 1) here.
    const auto countThreshold = nodeIdForCounts + 1;
    const auto leftCount = countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), leftValues);
    const auto rightCount = countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), rightValues);

    tree.createNode(modelbuilder::NodeData{splitColumnIndex, std::move(leftValues), std::move(rightValues), leftCount, rightCount});
  }

  // Depth tracking aligned to insertion path. LeftTree stores its own path privately, so we track here.
  // We push depth when we create a node (descend), and pop when LeftTree backtracks after leaf insertion.
  // To detect backtracking we can observe currentNodeID changes, but simplest is to recompute via tree.getCurrentNode().
  std::vector<Tree::Id> openNodeStack; // IDs of nodes on the current path after each node insertion
  std::vector<std::size_t> depthStack; // same size as openNodeStack

  auto currentDepth = [&]() -> std::size_t {
    if (depthStack.empty()) return 0;
    return depthStack.back();
  };

  while (!tree.complete()) {
    // Enforce max depth if requested.
    if (maxDepth != 0 && currentDepth() >= maxDepth) {
      tree.createLeaf(LeafData{0});
      // Leaf insertion may backtrack; update stacks below.
    } else {
      rebuildEnabledRowsForCurrentPosition();

      impl_->selector.findSignificantColumn();

      if (!impl_->selector.significantColumnFound() || !impl_->selector.significantPartitionFound()) {
        tree.createLeaf(LeafData{0});
      } else {
        const auto nodeIdForCounts = static_cast<std::uint32_t>(tree.currentNodeID() < 0 ? 0 : tree.currentNodeID());
        const auto splitColumnIndex = static_cast<std::uint64_t>(impl_->selector.getSignificantColumnIndex());
        // Partition "one" is left, partition "two" is right.
        auto leftValues = impl_->selector.getFirstPartition();
        auto rightValues = impl_->selector.getSecondPartition();
        const auto countThreshold = nodeIdForCounts + 1;
        const auto leftCount = countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), leftValues);
        const auto rightCount = countEnabledRowsAfterApplyingPartition(countThreshold, static_cast<std::size_t>(splitColumnIndex), rightValues);

        tree.createNode(modelbuilder::NodeData{splitColumnIndex, std::move(leftValues), std::move(rightValues), leftCount, rightCount});
      }
    }

    // Maintain a best-effort depth stack consistent with LeftTree's backtracking.
    // If current node is null, tree is complete.
    auto* cur = tree.getCurrentNode();
    if (!cur) break;

    // If we just created a node, it's the current node with no children yet.
    // Add it to the stack if not already.
    if (openNodeStack.empty() || openNodeStack.back() != cur->id) {
      // We only push when the current node has no children yet (newly created).
      if (!cur->left && !cur->right) {
        const std::size_t d = depthStack.empty() ? 1 : (depthStack.back() + 1);
        openNodeStack.push_back(cur->id);
        depthStack.push_back(d);
      } else {
        // Backtracked to an existing node; trim stacks to match.
        while (!openNodeStack.empty() && openNodeStack.back() != cur->id) {
          openNodeStack.pop_back();
          depthStack.pop_back();
        }
      }
    }
  }

  return tree;
}

std::uint64_t ModelBuilder::rowCount() const {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();
  return impl_->selector.getRowCount();
}

std::uint64_t ModelBuilder::columnCount() const {
  if (!impl_) throw std::logic_error("ModelBuilder: moved-from instance");
  impl_->ensureLoaded();
  return impl_->selector.getColumnCount();
}

namespace {

std::string escapeDotLabel(std::string s) {
  // Graphviz "label" strings support C-like escapes; simplest is to escape backslash and quotes.
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    // Graphviz labels use the two-character sequence "\n" to denote a line break.
    // We allow callers to use real newlines when building the label.
    if (ch == '\n') out += "\\n";
    else out.push_back(ch);
  }
  return out;
}

template <class T>
std::string vecToString(const std::vector<T>& v, std::size_t maxItems = 32) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < v.size() && i < maxItems; ++i) {
    if (i) oss << ",";
    oss << v[i];
  }
  if (v.size() > maxItems) oss << ",...";
  oss << "]";
  return oss.str();
}

} // namespace

void ModelBuilder::createGraphviz(const Tree& tree, const std::string& outputDotPath) const {
  std::ofstream out(outputDotPath);
  if (!out) {
    throw std::runtime_error("ModelBuilder::createGraphviz: unable to open output file: " + outputDotPath);
  }

  out << "digraph LeftTree {\n";
  out << "  rankdir=TB;\n";
  out << "  node [shape=box, fontname=\"Helvetica\"];\n";
  out << "  edge [fontname=\"Helvetica\"];\n";

  const auto* root = tree.root();
  if (!root) {
    out << "}\n";
    return;
  }

  // LeftTree uses sequential IDs: elementCount() is nextId_.
  for (std::size_t i = 0; i < tree.elementCount(); ++i) {
    const auto id = static_cast<Tree::Id>(i);
    auto type = tree.getElementType(id);
    if (!type.has_value()) continue;

    std::ostringstream label;
    label << "id=" << id;
    auto dataOpt = tree.getElementData(id);
    if (dataOpt.has_value()) {
      if (std::holds_alternative<Tree::NodeDataResult>(*dataOpt)) {
        const auto& [elemType, nd] = std::get<Tree::NodeDataResult>(*dataOpt);
        (void)elemType;
        label << "\nNode";
        if (nd.has_value()) {
          label << "\nsplitColumn=" << nd->splitColumnIndex();
          label << "\nleftCount=" << nd->leftPartitionCount();
          label << "\nleft=" << vecToString(nd->leftPartitionValues());
          label << "\nrightCount=" << nd->rightPartitionCount();
          label << "\nright=" << vecToString(nd->rightPartitionValues());
        } else {
          label << "\n(no data)";
        }
        out << "  n" << id << " [shape=box, label=\"" << escapeDotLabel(label.str()) << "\"];\n";
      } else {
        const auto& [elemType, lf] = std::get<Tree::LeafDataResult>(*dataOpt);
        (void)elemType;
        label << "\nLeaf";
        if (lf.has_value()) {
          label << "\nvalue=" << lf->placeholder();
        } else {
          label << "\n(no data)";
        }
        out << "  n" << id << " [shape=ellipse, label=\"" << escapeDotLabel(label.str()) << "\"];\n";
      }
    }
  }

  // Emit edges based on the stored structure.
  for (std::size_t i = 0; i < tree.elementCount(); ++i) {
    const auto id = static_cast<Tree::Id>(i);
    const auto* node = tree.getNode(id);
    if (!node) continue;
    if (node->left) {
      out << "  n" << id << " -> n" << node->left->id << " [label=\"L\"];\n";
    }
    if (node->right) {
      out << "  n" << id << " -> n" << node->right->id << " [label=\"R\"];\n";
    }
  }

  out << "}\n";
}

} // namespace modelbuilder

