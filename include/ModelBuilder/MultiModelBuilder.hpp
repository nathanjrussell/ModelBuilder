#pragma once

#include <ModelBuilder/TreeBuilder.hpp>

#include <cpp_type_concepts/Serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace modelbuilder {

// MultiModelBuilder: builds and stores many TreeBuilder artifacts.
//
// On-disk layout (big-endian, no versioning):
//   mb_trees.bin: repeated records
//       u64 treeLen
//       u8[treeLen] treeBytes (TreeBuilder::serialize output)
//   mb_map.bin: length+bytes wrapper
//       u64 payloadLen
//       payload = repeated pairs:
//           u64 targetColumn
//           u64 fileOffset   (byte offset into mb_trees.bin where the record's treeLen starts)
//
// The MultiModelBuilder object itself is Serializable and captures enough metadata to reopen the
// two files later and support lazy getTree(targetColumn).
class MultiModelBuilder {
public:
  MultiModelBuilder() = default;

  // Build all trees for the given target columns and write mb_trees.bin + mb_map.bin into outputDir.
  static MultiModelBuilder buildAndWrite(const std::string& parsedDir,
                                         const std::string& outputDir,
                                         const std::vector<std::size_t>& targetColumns,
                                         std::size_t maxDepth,
                                         double columnAlpha,
                                         bool columnAlphaApplyBonferroni,
                                         double partitionAlpha,
                                         bool partitionAlphaApplyBonferroni,
                                         const std::vector<std::size_t>& analysisColumns,
                                         std::size_t threadCount);

  [[nodiscard]] const std::string& parsedDir() const noexcept { return parsedDir_; }
  [[nodiscard]] const std::string& outputDir() const noexcept { return outputDir_; }

  [[nodiscard]] std::size_t maxDepth() const noexcept { return maxDepth_; }
  [[nodiscard]] double columnAlpha() const noexcept { return columnAlpha_; }
  [[nodiscard]] bool columnAlphaApplyBonferroni() const noexcept { return columnAlphaApplyBonferroni_; }
  [[nodiscard]] double partitionAlpha() const noexcept { return partitionAlpha_; }
  [[nodiscard]] bool partitionAlphaApplyBonferroni() const noexcept { return partitionAlphaApplyBonferroni_; }

  // Lazy-load a tree for the provided target column.
  [[nodiscard]] TreeBuilder getTree(std::uint64_t targetColumn) const;

  // Predict a probability distribution over target values for one sample.
  //
  // sample:
  //   Per-column encoded categorical values. Must have size >= max(splitColumn)+1 used by the tree.
  //
  // Missing value:
  //   A value of 0 is treated as missing.
  //   - if applyConditional == false: throws.
  //   - if applyConditional == true: traverses both branches and mixes probabilities with:
  //         p(left)  = leftCount  / (leftCount + rightCount)
  //         p(right) = rightCount / (leftCount + rightCount)
  //     This can occur multiple times in the same tree traversal.
  //
  // Errors:
  //   - If the model has not been built or loaded (no outputDir / map): throws.
  //   - If targetColumn does not exist: throws.
  //   - If a non-missing value is not present in either partition at a split: throws.
  [[nodiscard]] std::map<std::uint32_t, double> predict(const std::vector<std::uint64_t>& sample,
                                                       std::uint64_t targetColumn,
                                                       bool applyConditional) const;

  void serialize(std::ostream& out) const;
  static MultiModelBuilder deserialize(std::istream& in);

  // (Re)load mb_map.bin into memory. Called automatically by deserialize().
  void loadMap();

  [[nodiscard]] std::size_t treeCount() const noexcept { return targetToOffset_.size(); }

private:
  static std::string treesPath_(const std::string& outputDir);
  static std::string mapPath_(const std::string& outputDir);

  static void writeU8_(std::ostream& out, std::uint8_t v);
  static std::uint8_t readU8_(std::istream& in);
  static void writeU64BE_(std::ostream& out, std::uint64_t v);
  static std::uint64_t readU64BE_(std::istream& in);
  static void writeF64BE_(std::ostream& out, double d);
  static double readF64BE_(std::istream& in);
  static void writeBool_(std::ostream& out, bool b);
  static bool readBool_(std::istream& in);
  static void writeString_(std::ostream& out, const std::string& s);
  static std::string readString_(std::istream& in);

  std::string parsedDir_;
  std::string outputDir_;

  std::size_t maxDepth_ = 0;
  double columnAlpha_ = 0.05;
  bool columnAlphaApplyBonferroni_ = true;
  double partitionAlpha_ = 0.05;
  bool partitionAlphaApplyBonferroni_ = false;

  // targetColumn -> file offset in mb_trees.bin
  std::unordered_map<std::uint64_t, std::uint64_t> targetToOffset_;
};

static_assert(cpp_type_concepts::Serializable<MultiModelBuilder>);

} // namespace modelbuilder


