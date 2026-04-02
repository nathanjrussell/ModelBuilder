#include <ModelBuilder/MultiModelBuilder.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace modelbuilder {

namespace {
struct BuiltTree {
  std::uint64_t targetColumn = 0;
  std::string bytes; // serialized TreeBuilder payload
};

class BlockingQueue {
public:
  void push(BuiltTree v) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push(std::move(v));
    }
    cv_.notify_one();
  }

  // Returns false when closed and drained.
  bool pop(BuiltTree& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]() { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      closed_ = true;
    }
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<BuiltTree> q_;
  bool closed_ = false;
};

static std::vector<std::size_t> analysisExcludingTarget(const std::vector<std::size_t>& base,
                                                        std::size_t target) {
  std::vector<std::size_t> out;
  out.reserve(base.size());
  for (auto c : base) {
    if (c == target) continue;
    out.push_back(c);
  }
  return out;
}

} // namespace

std::string MultiModelBuilder::treesPath_(const std::string& outputDir) {
  return (std::filesystem::path{outputDir} / "mb_trees.bin").string();
}

std::string MultiModelBuilder::mapPath_(const std::string& outputDir) {
  return (std::filesystem::path{outputDir} / "mb_map.bin").string();
}

void MultiModelBuilder::writeU8_(std::ostream& out, std::uint8_t v) {
  out.put(static_cast<char>(v));
  if (!out) throw std::runtime_error("MultiModelBuilder: writeU8 failed");
}

std::uint8_t MultiModelBuilder::readU8_(std::istream& in) {
  const int c = in.get();
  if (c == std::char_traits<char>::eof()) throw std::runtime_error("MultiModelBuilder: readU8 failed");
  return static_cast<std::uint8_t>(c);
}

void MultiModelBuilder::writeU64BE_(std::ostream& out, std::uint64_t v) {
  for (int i = 7; i >= 0; --i) writeU8_(out, static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu));
}

std::uint64_t MultiModelBuilder::readU64BE_(std::istream& in) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | readU8_(in);
  return v;
}

void MultiModelBuilder::writeF64BE_(std::ostream& out, double d) {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(d);
  writeU64BE_(out, bits);
}

double MultiModelBuilder::readF64BE_(std::istream& in) {
  const std::uint64_t bits = readU64BE_(in);
  return std::bit_cast<double>(bits);
}

void MultiModelBuilder::writeBool_(std::ostream& out, bool b) { writeU8_(out, b ? 1u : 0u); }

bool MultiModelBuilder::readBool_(std::istream& in) {
  const auto v = readU8_(in);
  if (v != 0u && v != 1u) throw std::runtime_error("MultiModelBuilder: invalid boolean value");
  return v == 1u;
}

void MultiModelBuilder::writeString_(std::ostream& out, const std::string& s) {
  writeU64BE_(out, static_cast<std::uint64_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
  if (!out) throw std::runtime_error("MultiModelBuilder: writeString failed");
}

std::string MultiModelBuilder::readString_(std::istream& in) {
  const auto n = readU64BE_(in);
  std::string s;
  s.resize(static_cast<std::size_t>(n));
  if (n > 0) {
    in.read(s.data(), static_cast<std::streamsize>(s.size()));
    if (!in) throw std::runtime_error("MultiModelBuilder: readString failed");
  }
  return s;
}

MultiModelBuilder MultiModelBuilder::buildAndWrite(const std::string& parsedDir,
                                                   const std::string& outputDir,
                                                   const std::vector<std::size_t>& targetColumns,
                                                   std::size_t maxDepth,
                                                   double columnAlpha,
                                                   bool columnAlphaApplyBonferroni,
                                                   double partitionAlpha,
                                                   bool partitionAlphaApplyBonferroni,
                                                   const std::vector<std::size_t>& analysisColumns,
                                                   std::size_t threadCount) {
  if (threadCount == 0) threadCount = 1;

  std::filesystem::create_directories(outputDir);

  const auto treesPath = treesPath_(outputDir);
  std::ofstream treesOut(treesPath, std::ios::binary | std::ios::trunc);
  if (!treesOut) throw std::runtime_error("MultiModelBuilder: failed to open mb_trees.bin for write");

  BlockingQueue q;
  std::atomic<std::size_t> next{0};

  // Writer state
  std::mutex mapMu;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> mapPairs;
  mapPairs.reserve(targetColumns.size());

  auto writer = std::thread([&]() {
    BuiltTree item;
    while (q.pop(item)) {
      const std::uint64_t off = static_cast<std::uint64_t>(treesOut.tellp());
      // record: u64 len + bytes
      writeU64BE_(treesOut, static_cast<std::uint64_t>(item.bytes.size()));
      if (!item.bytes.empty()) {
        treesOut.write(item.bytes.data(), static_cast<std::streamsize>(item.bytes.size()));
      }
      if (!treesOut) throw std::runtime_error("MultiModelBuilder: failed writing mb_trees.bin");

      {
        std::lock_guard<std::mutex> lk(mapMu);
        mapPairs.emplace_back(item.targetColumn, off);
      }
    }
  });

  auto workerFn = [&]() {
    while (true) {
      const auto i = next.fetch_add(1);
      if (i >= targetColumns.size()) break;
      const auto tgt = targetColumns[i];

      const auto perTreeAnalysis = analysisExcludingTarget(analysisColumns, tgt);

      auto artifact = TreeBuilder::buildFromDataDir(parsedDir,
                                                   tgt,
                                                   maxDepth,
                                                   columnAlpha,
                                                   columnAlphaApplyBonferroni,
                                                   partitionAlpha,
                                                   partitionAlphaApplyBonferroni,
                                                   perTreeAnalysis);

      std::ostringstream tmp(std::ios::binary);
      artifact.serialize(tmp);
      q.push(BuiltTree{static_cast<std::uint64_t>(tgt), tmp.str()});
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(threadCount);
  for (std::size_t t = 0; t < threadCount; ++t) workers.emplace_back(workerFn);

  for (auto& th : workers) th.join();
  q.close();
  writer.join();

  // Sort by target column for deterministic output, and validate uniqueness.
  std::sort(mapPairs.begin(), mapPairs.end(), [](auto& a, auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
  });

  for (std::size_t i = 1; i < mapPairs.size(); ++i) {
    if (mapPairs[i - 1].first == mapPairs[i].first) {
      throw std::runtime_error("MultiModelBuilder: duplicate targetColumn in map");
    }
  }

  // Write mb_map.bin as length+bytes.
  const auto mapPath = mapPath_(outputDir);
  std::ofstream mapOut(mapPath, std::ios::binary | std::ios::trunc);
  if (!mapOut) throw std::runtime_error("MultiModelBuilder: failed to open mb_map.bin for write");

  const std::uint64_t payloadLen = static_cast<std::uint64_t>(mapPairs.size()) * 16ull;
  writeU64BE_(mapOut, payloadLen);
  for (const auto& [tc, off] : mapPairs) {
    writeU64BE_(mapOut, tc);
    writeU64BE_(mapOut, off);
  }
  if (!mapOut) throw std::runtime_error("MultiModelBuilder: failed writing mb_map.bin");

  MultiModelBuilder mb;
  mb.parsedDir_ = parsedDir;
  mb.outputDir_ = outputDir;
  mb.maxDepth_ = maxDepth;
  mb.columnAlpha_ = columnAlpha;
  mb.columnAlphaApplyBonferroni_ = columnAlphaApplyBonferroni;
  mb.partitionAlpha_ = partitionAlpha;
  mb.partitionAlphaApplyBonferroni_ = partitionAlphaApplyBonferroni;
  for (const auto& [tc, off] : mapPairs) mb.targetToOffset_.emplace(tc, off);

  return mb;
}

void MultiModelBuilder::loadMap() {
  targetToOffset_.clear();

  const auto mapPath = mapPath_(outputDir_);
  std::ifstream in(mapPath, std::ios::binary);
  if (!in) throw std::runtime_error("MultiModelBuilder: failed to open mb_map.bin");

  const std::uint64_t payloadLen = readU64BE_(in);
  if (payloadLen % 16ull != 0ull) throw std::runtime_error("MultiModelBuilder: invalid mb_map.bin payload length");

  const std::uint64_t pairCount = payloadLen / 16ull;
  for (std::uint64_t i = 0; i < pairCount; ++i) {
    const auto tc = readU64BE_(in);
    const auto off = readU64BE_(in);
    targetToOffset_.emplace(tc, off);
  }

  if (!in) throw std::runtime_error("MultiModelBuilder: failed reading mb_map.bin");
}

TreeBuilder MultiModelBuilder::getTree(std::uint64_t targetColumn) const {
  const auto it = targetToOffset_.find(targetColumn);
  if (it == targetToOffset_.end()) {
    throw std::out_of_range("MultiModelBuilder::getTree: target column not found");
  }

  const auto treesPath = treesPath_(outputDir_);
  std::ifstream in(treesPath, std::ios::binary);
  if (!in) throw std::runtime_error("MultiModelBuilder::getTree: failed to open mb_trees.bin");

  in.seekg(static_cast<std::streamoff>(it->second));
  if (!in) throw std::runtime_error("MultiModelBuilder::getTree: failed to seek to tree offset");

  const std::uint64_t len = readU64BE_(in);
  std::string blob;
  blob.resize(static_cast<std::size_t>(len));
  if (len > 0) {
    in.read(blob.data(), static_cast<std::streamsize>(blob.size()));
    if (!in) throw std::runtime_error("MultiModelBuilder::getTree: failed reading tree bytes");
  }

  std::istringstream tmp(blob, std::ios::binary);
  return TreeBuilder::deserialize(tmp);
}

namespace {
using Dist = std::map<std::uint32_t, double>;

static Dist addWeighted(const Dist& a, double wa, const Dist& b, double wb) {
  Dist out;
  for (const auto& [k, v] : a) out[k] += wa * v;
  for (const auto& [k, v] : b) out[k] += wb * v;
  return out;
}

static Dist normalizeCountsOrThrow(const std::map<std::uint32_t, std::uint64_t>& counts) {
  std::uint64_t total = 0;
  for (const auto& kv : counts) total += kv.second;
  if (total == 0) throw std::runtime_error("MultiModelBuilder::predict: leaf has zero total targetCounts");

  Dist out;
  const double denom = static_cast<double>(total);
  for (const auto& [k, c] : counts) out.emplace(k, static_cast<double>(c) / denom);
  return out;
}

static bool containsU32(const std::vector<std::uint32_t>& v, std::uint64_t needleU64) {
  const auto needle = static_cast<std::uint32_t>(needleU64);
  return std::find(v.begin(), v.end(), needle) != v.end();
}

static Dist predictFromIdOrThrow(const modelbuilder::TreeBuilder::Tree& tree,
                                 std::size_t id,
                                 const std::vector<std::uint64_t>& sample,
                                 bool applyConditional) {
  auto dataOpt = tree.getElementData(id);
  if (!dataOpt.has_value()) throw std::runtime_error("MultiModelBuilder::predict: missing element data");

  if (std::holds_alternative<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*dataOpt)) {
    const auto& [t, leafOpt] = std::get<modelbuilder::TreeBuilder::Tree::LeafDataResult>(*dataOpt);
    (void)t;
    if (!leafOpt.has_value()) throw std::runtime_error("MultiModelBuilder::predict: leaf has no payload");
    return normalizeCountsOrThrow(leafOpt->targetCounts());
  }

  if (!std::holds_alternative<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*dataOpt)) {
    throw std::runtime_error("MultiModelBuilder::predict: unexpected element type");
  }

  const auto& [t, nodeOpt] = std::get<modelbuilder::TreeBuilder::Tree::NodeDataResult>(*dataOpt);
  (void)t;
  if (!nodeOpt.has_value()) throw std::runtime_error("MultiModelBuilder::predict: node has no payload");
  const auto& nd = *nodeOpt;

  const auto split = static_cast<std::size_t>(nd.splitColumnIndex());
  if (split >= sample.size()) {
    throw std::runtime_error("MultiModelBuilder::predict: sample is missing split column index");
  }

  const std::uint64_t v = sample[split];

  const auto* node = tree.getNode(static_cast<typename modelbuilder::TreeBuilder::Tree::Id>(id));
  if (!node) throw std::runtime_error("MultiModelBuilder::predict: failed to get node by id");

  if (v == 0) {
    if (!applyConditional) {
      throw std::runtime_error("MultiModelBuilder::predict: encountered missing value (0) but applyConditional=false");
    }
    const std::uint64_t leftCount = nd.leftPartitionCount();
    const std::uint64_t rightCount = nd.rightPartitionCount();
    const std::uint64_t denomU = leftCount + rightCount;
    if (denomU == 0) {
      throw std::runtime_error("MultiModelBuilder::predict: conditional weights undefined (left+right==0)");
    }
    if (!node->left || !node->right) {
      throw std::runtime_error("MultiModelBuilder::predict: conditional traversal requires both children");
    }
    const double denom = static_cast<double>(denomU);
    const double pL = static_cast<double>(leftCount) / denom;
    const double pR = static_cast<double>(rightCount) / denom;
    const auto dl = predictFromIdOrThrow(tree, static_cast<std::size_t>(node->left->id), sample, applyConditional);
    const auto dr = predictFromIdOrThrow(tree, static_cast<std::size_t>(node->right->id), sample, applyConditional);
    return addWeighted(dl, pL, dr, pR);
  }

  const bool inLeft = containsU32(nd.leftPartitionValues(), v);
  const bool inRight = containsU32(nd.rightPartitionValues(), v);
  if (inLeft == inRight) {
    // Either in neither, or (unexpectedly) in both.
    throw std::runtime_error("MultiModelBuilder::predict: value not in exactly one partition for split");
  }

  if (inLeft) {
    if (!node->left) throw std::runtime_error("MultiModelBuilder::predict: missing left child");
    return predictFromIdOrThrow(tree, static_cast<std::size_t>(node->left->id), sample, applyConditional);
  }
  if (!node->right) throw std::runtime_error("MultiModelBuilder::predict: missing right child");
  return predictFromIdOrThrow(tree, static_cast<std::size_t>(node->right->id), sample, applyConditional);
}

} // namespace


std::map<std::uint32_t, double> MultiModelBuilder::predict(const std::vector<std::uint64_t>& sample,
                                                          std::uint64_t targetColumn,
                                                          bool applyConditional) const {
  if (outputDir_.empty()) {
    throw std::runtime_error("MultiModelBuilder::predict: model not initialized (outputDir is empty)");
  }
  if (targetToOffset_.empty()) {
    throw std::runtime_error("MultiModelBuilder::predict: model lookup map is empty (buildAndWrite/deserialize/loadMap required)");
  }

  auto artifact = getTree(targetColumn); // throws if missing
  const auto& tree = artifact.tree();
  if (tree.root() == nullptr || tree.elementCount() == 0) {
    throw std::runtime_error("MultiModelBuilder::predict: tree is empty");
  }

  auto dist = predictFromIdOrThrow(tree, /*id=*/0, sample, applyConditional);

  // Final renormalization to absorb floating-error drift from repeated branching.
  double s = 0.0;
  for (const auto& kv : dist) s += kv.second;
  if (!(s > 0.0)) throw std::runtime_error("MultiModelBuilder::predict: invalid probability mass");
  for (auto& kv : dist) kv.second /= s;
  return dist;
}

void MultiModelBuilder::serialize(std::ostream& out) const {
  // length+bytes overall not required here; we just provide a deterministic binary encoding.
  writeString_(out, parsedDir_);
  writeString_(out, outputDir_);
  writeU64BE_(out, static_cast<std::uint64_t>(maxDepth_));
  writeF64BE_(out, columnAlpha_);
  writeBool_(out, columnAlphaApplyBonferroni_);
  writeF64BE_(out, partitionAlpha_);
  writeBool_(out, partitionAlphaApplyBonferroni_);

  // Store target->offset map in the object serialization as well (so deserialize does not *require*
  // reading mb_map.bin first), but still keep loadMap() available.
  writeU64BE_(out, static_cast<std::uint64_t>(targetToOffset_.size()));
  // Serialize in sorted order for determinism.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
  pairs.reserve(targetToOffset_.size());
  for (const auto& kv : targetToOffset_) pairs.emplace_back(kv.first, kv.second);
  std::sort(pairs.begin(), pairs.end());
  for (const auto& [tc, off] : pairs) {
    writeU64BE_(out, tc);
    writeU64BE_(out, off);
  }

  if (!out) throw std::runtime_error("MultiModelBuilder::serialize failed");
}

MultiModelBuilder MultiModelBuilder::deserialize(std::istream& in) {
  MultiModelBuilder mb;
  mb.parsedDir_ = readString_(in);
  mb.outputDir_ = readString_(in);
  mb.maxDepth_ = static_cast<std::size_t>(readU64BE_(in));
  mb.columnAlpha_ = readF64BE_(in);
  mb.columnAlphaApplyBonferroni_ = readBool_(in);
  mb.partitionAlpha_ = readF64BE_(in);
  mb.partitionAlphaApplyBonferroni_ = readBool_(in);

  const std::uint64_t n = readU64BE_(in);
  for (std::uint64_t i = 0; i < n; ++i) {
    const auto tc = readU64BE_(in);
    const auto off = readU64BE_(in);
    mb.targetToOffset_.emplace(tc, off);
  }

  if (!in) throw std::runtime_error("MultiModelBuilder::deserialize failed");
  return mb;
}

MultiModelBuilder MultiModelBuilder::open(const std::string& outputDir) {
  MultiModelBuilder mb;
  mb.outputDir_ = outputDir;
  mb.loadMap();
  return mb;
}

} // namespace modelbuilder

