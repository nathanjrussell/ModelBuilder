#pragma once

#include <cpp_type_concepts/Serializable.hpp>

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace modelbuilder {

// Trivial leaf payload placeholder. Will be extended later.
// Serialization format (big-endian, no versioning): u32 placeholder.
class LeafData {
public:
  LeafData() = default;
  explicit LeafData(std::uint32_t placeholder) : placeholder_(placeholder) {}

  [[nodiscard]] std::uint32_t placeholder() const noexcept { return placeholder_; }

  void serialize(std::ostream& out) const {
    out.put(static_cast<char>((placeholder_ >> 24) & 0xFFu));
    out.put(static_cast<char>((placeholder_ >> 16) & 0xFFu));
    out.put(static_cast<char>((placeholder_ >> 8) & 0xFFu));
    out.put(static_cast<char>(placeholder_ & 0xFFu));
    if (!out) throw std::runtime_error("LeafData::serialize failed");
  }

  static LeafData deserialize(std::istream& in) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const int c = in.get();
      if (c == std::char_traits<char>::eof()) throw std::runtime_error("LeafData::deserialize failed");
      v = (v << 8) | static_cast<std::uint32_t>(static_cast<unsigned char>(c));
    }
    return LeafData{v};
  }

private:
  std::uint32_t placeholder_ = 0;
};

static_assert(cpp_type_concepts::Serializable<LeafData>);

} // namespace modelbuilder

