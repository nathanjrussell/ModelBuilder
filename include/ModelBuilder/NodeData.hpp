#pragma once

#include <cpp_type_concepts/Serializable.hpp>

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace modelbuilder {

// Payload stored in LeftTree node elements.
//
// Counts are branch row counts:
//   - leftPartitionCount  == number of enabled rows that reach the left child
//   - rightPartitionCount == number of enabled rows that reach the right child
//
// Serialization format (big-endian, no versioning):
//   u64 splitColumnIndex
//   u64 leftPartitionCount
//   u64 rightPartitionCount
//   u64 leftPartitionValues.size
//   u32[leftSize] leftPartitionValues
//   u64 rightPartitionValues.size
//   u32[rightSize] rightPartitionValues
class NodeData {
public:
  NodeData() = default;

  NodeData(std::uint64_t splitColumnIndex,
           std::vector<std::uint32_t> leftPartitionValues,
           std::vector<std::uint32_t> rightPartitionValues,
           std::uint64_t leftPartitionCount,
           std::uint64_t rightPartitionCount)
      : splitColumnIndex_(splitColumnIndex),
        leftPartitionValues_(std::move(leftPartitionValues)),
        rightPartitionValues_(std::move(rightPartitionValues)),
        leftPartitionCount_(leftPartitionCount),
        rightPartitionCount_(rightPartitionCount) {}

  [[nodiscard]] std::uint64_t splitColumnIndex() const noexcept { return splitColumnIndex_; }
  [[nodiscard]] const std::vector<std::uint32_t>& leftPartitionValues() const noexcept { return leftPartitionValues_; }
  [[nodiscard]] const std::vector<std::uint32_t>& rightPartitionValues() const noexcept { return rightPartitionValues_; }
  [[nodiscard]] std::uint64_t leftPartitionCount() const noexcept { return leftPartitionCount_; }
  [[nodiscard]] std::uint64_t rightPartitionCount() const noexcept { return rightPartitionCount_; }

  void serialize(std::ostream& out) const {
    writeU64BE_(out, splitColumnIndex_);
    writeU64BE_(out, leftPartitionCount_);
    writeU64BE_(out, rightPartitionCount_);

    writeU64BE_(out, static_cast<std::uint64_t>(leftPartitionValues_.size()));
    for (std::uint32_t v : leftPartitionValues_) writeU32BE_(out, v);

    writeU64BE_(out, static_cast<std::uint64_t>(rightPartitionValues_.size()));
    for (std::uint32_t v : rightPartitionValues_) writeU32BE_(out, v);

    if (!out) throw std::runtime_error("NodeData::serialize failed");
  }

  static NodeData deserialize(std::istream& in) {
    NodeData nd;
    nd.splitColumnIndex_ = readU64BE_(in);
    nd.leftPartitionCount_ = readU64BE_(in);
    nd.rightPartitionCount_ = readU64BE_(in);

    const std::uint64_t leftN = readU64BE_(in);
    nd.leftPartitionValues_.resize(static_cast<std::size_t>(leftN));
    for (std::uint64_t i = 0; i < leftN; ++i) {
      nd.leftPartitionValues_[static_cast<std::size_t>(i)] = readU32BE_(in);
    }

    const std::uint64_t rightN = readU64BE_(in);
    nd.rightPartitionValues_.resize(static_cast<std::size_t>(rightN));
    for (std::uint64_t i = 0; i < rightN; ++i) {
      nd.rightPartitionValues_[static_cast<std::size_t>(i)] = readU32BE_(in);
    }

    return nd;
  }

private:
  static void writeU8_(std::ostream& out, std::uint8_t v) {
    out.put(static_cast<char>(v));
    if (!out) throw std::runtime_error("writeU8 failed");
  }

  static std::uint8_t readU8_(std::istream& in) {
    const int c = in.get();
    if (c == std::char_traits<char>::eof()) throw std::runtime_error("readU8 failed");
    return static_cast<std::uint8_t>(c);
  }

  static void writeU32BE_(std::ostream& out, std::uint32_t v) {
    for (int i = 3; i >= 0; --i) {
      writeU8_(out, static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
    }
  }

  static std::uint32_t readU32BE_(std::istream& in) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      v = (v << 8) | readU8_(in);
    }
    return v;
  }

  static void writeU64BE_(std::ostream& out, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) {
      writeU8_(out, static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
    }
  }

  static std::uint64_t readU64BE_(std::istream& in) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v = (v << 8) | readU8_(in);
    }
    return v;
  }

  std::uint64_t splitColumnIndex_ = 0;
  std::vector<std::uint32_t> leftPartitionValues_;
  std::vector<std::uint32_t> rightPartitionValues_;
  std::uint64_t leftPartitionCount_ = 0;
  std::uint64_t rightPartitionCount_ = 0;
};

static_assert(cpp_type_concepts::Serializable<NodeData>);

} // namespace modelbuilder

