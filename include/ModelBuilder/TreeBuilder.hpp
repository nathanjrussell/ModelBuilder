#pragma once

#include <LeftTree/LeftTree.hpp>
#include <LeftTree/LeftTreeSerialization.hpp>

#include <ModelBuilder/LeafData.hpp>
#include <ModelBuilder/NodeData.hpp>

#include <cpp_type_concepts/Serializable.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace modelbuilder {

// A serializable, self-contained single-tree “artifact”.
//
// IMPORTANT: analysis/split-candidate columns are build-time only and are NOT stored or serialized.
//
// Serialization format (big-endian, no versioning):
//   u64 targetColumn
//   u64 maxDepth
//   f64 columnAlpha
//   u8  columnAlphaApplyBonferroni (0 or 1)
//   f64 partitionAlpha
//   u8  partitionAlphaApplyBonferroni (0 or 1)
//   u64 treeBlobLen
//   u8[treeBlobLen] treeBlob (exact bytes from LeftTreeSerialization::serializeToStream)
class TreeBuilder {
public:
  using Tree = lefttree::LeftTree<modelbuilder::NodeData, modelbuilder::LeafData>;
  using TreeSer = lefttree::LeftTreeSerialization<modelbuilder::NodeData, modelbuilder::LeafData>;

  // Build a serializable artifact from a parsed DataTable output directory.
  //
  // analysisColumns:
  //   Optional list of split candidate columns to enable during training.
  //   This is NOT stored in the returned artifact.
  static TreeBuilder buildFromDataDir(const std::string& parsedDir,
                                     std::size_t targetColumn,
                                     std::size_t maxDepth,
                                     double columnAlpha,
                                     bool columnAlphaApplyBonferroni,
                                     double partitionAlpha,
                                     bool partitionAlphaApplyBonferroni,
                                     const std::vector<std::size_t>& analysisColumns = {});

  TreeBuilder() = default;

  explicit TreeBuilder(std::size_t targetColumn,
                       std::size_t maxDepth,
                       double columnAlpha,
                       bool columnAlphaApplyBonferroni,
                       double partitionAlpha,
                       bool partitionAlphaApplyBonferroni,
                       Tree tree)
      : targetColumn_(targetColumn),
        maxDepth_(maxDepth),
        columnAlpha_(columnAlpha),
        columnAlphaApplyBonferroni_(columnAlphaApplyBonferroni),
        partitionAlpha_(partitionAlpha),
        partitionAlphaApplyBonferroni_(partitionAlphaApplyBonferroni),
        tree_(std::move(tree)) {}

  [[nodiscard]] std::size_t targetColumn() const noexcept { return targetColumn_; }
  [[nodiscard]] std::size_t maxDepth() const noexcept { return maxDepth_; }
  [[nodiscard]] double columnAlpha() const noexcept { return columnAlpha_; }
  [[nodiscard]] bool columnAlphaApplyBonferroni() const noexcept { return columnAlphaApplyBonferroni_; }
  [[nodiscard]] double partitionAlpha() const noexcept { return partitionAlpha_; }
  [[nodiscard]] bool partitionAlphaApplyBonferroni() const noexcept { return partitionAlphaApplyBonferroni_; }
  [[nodiscard]] const Tree& tree() const noexcept { return tree_; }

  void serialize(std::ostream& out) const {
    writeU64BE_(out, static_cast<std::uint64_t>(targetColumn_));
    writeU64BE_(out, static_cast<std::uint64_t>(maxDepth_));
    writeF64BE_(out, columnAlpha_);
    writeBool_(out, columnAlphaApplyBonferroni_);
    writeF64BE_(out, partitionAlpha_);
    writeBool_(out, partitionAlphaApplyBonferroni_);

    std::ostringstream tmp(std::ios::binary);
    TreeSer::serializeToStream(tree_, tmp);
    const std::string blob = tmp.str();
    writeU64BE_(out, static_cast<std::uint64_t>(blob.size()));
    out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    if (!out) throw std::runtime_error("TreeBuilder::serialize failed");
  }

  static TreeBuilder deserialize(std::istream& in) {
    const auto target = static_cast<std::size_t>(readU64BE_(in));
    const auto depth = static_cast<std::size_t>(readU64BE_(in));
    const double colAlpha = readF64BE_(in);
    const bool colBonf = readBool_(in);
    const double partAlpha = readF64BE_(in);
    const bool partBonf = readBool_(in);

    const std::uint64_t blobLen = readU64BE_(in);
    std::string blob;
    blob.resize(static_cast<std::size_t>(blobLen));
    if (blobLen > 0) {
      in.read(blob.data(), static_cast<std::streamsize>(blob.size()));
      if (!in) throw std::runtime_error("TreeBuilder::deserialize failed reading tree blob");
    }
    std::istringstream tmp(blob, std::ios::binary);
    Tree t = TreeSer::deserializeFromStream(tmp);

    return TreeBuilder{target, depth, colAlpha, colBonf, partAlpha, partBonf, std::move(t)};
  }

private:
  static void writeU8_(std::ostream& out, std::uint8_t v) {
    out.put(static_cast<char>(v));
    if (!out) throw std::runtime_error("TreeBuilder: writeU8 failed");
  }

  static std::uint8_t readU8_(std::istream& in) {
    const int c = in.get();
    if (c == std::char_traits<char>::eof()) throw std::runtime_error("TreeBuilder: readU8 failed");
    return static_cast<std::uint8_t>(c);
  }

  static void writeU64BE_(std::ostream& out, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) writeU8_(out, static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
  }

  static std::uint64_t readU64BE_(std::istream& in) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | readU8_(in);
    return v;
  }

  static void writeF64BE_(std::ostream& out, double d) {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(d);
    writeU64BE_(out, bits);
  }

  static double readF64BE_(std::istream& in) {
    const std::uint64_t bits = readU64BE_(in);
    return std::bit_cast<double>(bits);
  }

  static void writeBool_(std::ostream& out, bool b) { writeU8_(out, b ? 1u : 0u); }

  static bool readBool_(std::istream& in) {
    const auto v = readU8_(in);
    if (v != 0u && v != 1u) throw std::runtime_error("TreeBuilder: invalid boolean value");
    return v == 1u;
  }

  std::size_t targetColumn_ = 0;
  std::size_t maxDepth_ = 0;
  double columnAlpha_ = 0.05;
  bool columnAlphaApplyBonferroni_ = true;
  double partitionAlpha_ = 0.05;
  bool partitionAlphaApplyBonferroni_ = false;
  Tree tree_;
};

static_assert(cpp_type_concepts::Serializable<TreeBuilder>);

} // namespace modelbuilder

