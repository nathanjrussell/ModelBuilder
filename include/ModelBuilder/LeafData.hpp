#pragma once

#include <cpp_type_concepts/Serializable.hpp>

#include <cstdint>
#include <map>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace modelbuilder {

// Leaf payload.
//
// A leaf can happen for two reasons:
//  - NoSignificantColumn: FeatureSelector did not find any significant split candidate.
//  - NoSignificantPartition: A significant column was found but no significant partition.
//
// If a significant column was found, splitColumnIndex is that column index. Otherwise it is -1.
//
// Serialization format (big-endian, no versioning):
//   u8  reason
//   i64 splitColumnIndexOrMinusOne
//   u64 targetCounts.size
//   repeat targetCounts.size times:
//      u32 targetValue
//      u64 count
class LeafData {
public:
  enum class Reason : std::uint8_t {
    NoSignificantColumn = 0,
    NoSignificantPartition = 1,
  };

  LeafData() = default;

  LeafData(Reason reason,
           std::int64_t splitColumnIndexOrMinusOne,
           std::map<std::uint32_t, std::uint64_t> targetCounts)
      : reason_(reason),
        splitColumnIndexOrMinusOne_(splitColumnIndexOrMinusOne),
        targetCounts_(std::move(targetCounts)) {}

  static LeafData noSignificantColumn(std::map<std::uint32_t, std::uint64_t> targetCounts) {
    return LeafData{Reason::NoSignificantColumn, -1, std::move(targetCounts)};
  }
  static LeafData noSignificantPartition(std::size_t significantColumnIndex,
                                        std::map<std::uint32_t, std::uint64_t> targetCounts) {
    return LeafData{Reason::NoSignificantPartition,
                    static_cast<std::int64_t>(significantColumnIndex),
                    std::move(targetCounts)};
  }

  [[nodiscard]] Reason reason() const noexcept { return reason_; }
  [[nodiscard]] std::int64_t splitColumnIndexOrMinusOne() const noexcept { return splitColumnIndexOrMinusOne_; }
  [[nodiscard]] const std::map<std::uint32_t, std::uint64_t>& targetCounts() const noexcept { return targetCounts_; }

  void serialize(std::ostream& out) const {
    out.put(static_cast<char>(static_cast<std::uint8_t>(reason_)));

    const auto u = static_cast<std::uint64_t>(splitColumnIndexOrMinusOne_);
    out.put(static_cast<char>((u >> 56) & 0xFFu));
    out.put(static_cast<char>((u >> 48) & 0xFFu));
    out.put(static_cast<char>((u >> 40) & 0xFFu));
    out.put(static_cast<char>((u >> 32) & 0xFFu));
    out.put(static_cast<char>((u >> 24) & 0xFFu));
    out.put(static_cast<char>((u >> 16) & 0xFFu));
    out.put(static_cast<char>((u >> 8) & 0xFFu));
    out.put(static_cast<char>(u & 0xFFu));

    const auto n = static_cast<std::uint64_t>(targetCounts_.size());
    out.put(static_cast<char>((n >> 56) & 0xFFu));
    out.put(static_cast<char>((n >> 48) & 0xFFu));
    out.put(static_cast<char>((n >> 40) & 0xFFu));
    out.put(static_cast<char>((n >> 32) & 0xFFu));
    out.put(static_cast<char>((n >> 24) & 0xFFu));
    out.put(static_cast<char>((n >> 16) & 0xFFu));
    out.put(static_cast<char>((n >> 8) & 0xFFu));
    out.put(static_cast<char>(n & 0xFFu));

    for (const auto& [k, v] : targetCounts_) {
      out.put(static_cast<char>((k >> 24) & 0xFFu));
      out.put(static_cast<char>((k >> 16) & 0xFFu));
      out.put(static_cast<char>((k >> 8) & 0xFFu));
      out.put(static_cast<char>(k & 0xFFu));

      out.put(static_cast<char>((v >> 56) & 0xFFu));
      out.put(static_cast<char>((v >> 48) & 0xFFu));
      out.put(static_cast<char>((v >> 40) & 0xFFu));
      out.put(static_cast<char>((v >> 32) & 0xFFu));
      out.put(static_cast<char>((v >> 24) & 0xFFu));
      out.put(static_cast<char>((v >> 16) & 0xFFu));
      out.put(static_cast<char>((v >> 8) & 0xFFu));
      out.put(static_cast<char>(v & 0xFFu));
    }
    if (!out) throw std::runtime_error("LeafData::serialize failed");
  }

  static LeafData deserialize(std::istream& in) {
    const int rc = in.get();
    if (rc == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
    const auto reason = static_cast<Reason>(static_cast<std::uint8_t>(static_cast<unsigned char>(rc)));

    std::uint64_t u = 0;
    for (int i = 0; i < 8; ++i) {
      const int c = in.get();
      if (c == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
      u = (u << 8) | static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    }
    const auto idx = static_cast<std::int64_t>(u);

    std::uint64_t n = 0;
    for (int i = 0; i < 8; ++i) {
      const int c = in.get();
      if (c == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
      n = (n << 8) | static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    }

    std::map<std::uint32_t, std::uint64_t> counts;
    for (std::uint64_t i = 0; i < n; ++i) {
      std::uint32_t k = 0;
      for (int j = 0; j < 4; ++j) {
        const int c = in.get();
        if (c == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
        k = (k << 8) | static_cast<std::uint32_t>(static_cast<unsigned char>(c));
      }

      std::uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        const int c = in.get();
        if (c == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
        v = (v << 8) | static_cast<std::uint64_t>(static_cast<unsigned char>(c));
      }
      counts.emplace(k, v);
    }

    return LeafData{reason, idx, std::move(counts)};
  }

private:
  Reason reason_ = Reason::NoSignificantColumn;
  std::int64_t splitColumnIndexOrMinusOne_ = -1;
  std::map<std::uint32_t, std::uint64_t> targetCounts_;
};

static_assert(cpp_type_concepts::Serializable<LeafData>);

} // namespace modelbuilder

