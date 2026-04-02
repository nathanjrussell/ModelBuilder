#include <ModelBuilder/MultiModelBuilder.hpp>

#include <DataTable/DataTable.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  // Identical to ModelBuilder_example.cpp, but limits the number of target/analysis columns so it
  // runs quickly during development.
  //
  // Usage:
  //   ./ModelBuilder_example_fast [--reparse]

  const std::filesystem::path csvPath = std::filesystem::path{"examples"} / "sample_data" / "ptsd.csv";
  const std::filesystem::path parsedDir = std::filesystem::path{"build"} / "datasets" / "ptsd_output";
  const std::filesystem::path modelOutDir = std::filesystem::path{"build"} / "models" / "ptsd_model_fast";

  const bool forceReparse = (argc > 1 && std::string_view(argv[1]) == "--reparse");

  try {
    DataTableLib::DataTable dt;
    const auto metaDir = parsedDir / "meta_data";
    if (!forceReparse && std::filesystem::exists(metaDir)) {
      dt.load(parsedDir.string());
      std::cout << "Loaded parsed dataset: " << parsedDir << "\n";
    } else {
      dt.setInputFilePath(csvPath.string());
      dt.setOutputDirectory(parsedDir.string());
      dt.parse(/*threads=*/1);
      std::cout << "Parsed CSV: " << csvPath << " -> " << parsedDir << "\n";
    }

    const std::size_t cols = static_cast<std::size_t>(dt.getColumnCount());
    if (cols < 3) throw std::runtime_error("Need at least 3 columns (id + >=2 features)");

    // Parameters (same defaults as the full example).
    const std::size_t maxDepth = 0; // unlimited
    const double columnAlpha = 0.005;
    const bool columnAlphaBonferroni = true;
    const double partitionAlpha = 0.005;
    const bool partitionAlphaBonferroni = false;

    // analysisColumns: all columns except column 0 (id), but we cap it for speed.
    // IMPORTANT: include the chosen wantTarget so that targetColumns contains it.
    const std::uint64_t wantTarget = 1;
    const std::size_t maxEnabledColumns = 12; // tune this for speed vs coverage

    std::vector<std::size_t> analysisColumns;
    analysisColumns.reserve(cols);
    for (std::size_t c = 0; c < cols; ++c) {
      if (c == 0) continue;
      analysisColumns.push_back(c);
    }
    if (analysisColumns.size() > maxEnabledColumns) analysisColumns.resize(maxEnabledColumns);

    if (std::find(analysisColumns.begin(), analysisColumns.end(), static_cast<std::size_t>(wantTarget)) ==
        analysisColumns.end()) {
      analysisColumns.insert(analysisColumns.begin(), static_cast<std::size_t>(wantTarget));
      if (analysisColumns.size() > maxEnabledColumns) analysisColumns.resize(maxEnabledColumns);
    }

    // targetColumns: keep a small subset (first N analysis columns).
    // Per-tree build will remove the target column from analysis columns.
    std::vector<std::size_t> targetColumns = analysisColumns;

    const std::size_t threadCount = 8;

    auto multi = modelbuilder::MultiModelBuilder::buildAndWrite(parsedDir.string(),
                                                                modelOutDir.string(),
                                                                targetColumns,
                                                                maxDepth,
                                                                columnAlpha,
                                                                columnAlphaBonferroni,
                                                                partitionAlpha,
                                                                partitionAlphaBonferroni,
                                                                analysisColumns,
                                                                threadCount);

    std::cout << "Built " << multi.treeCount() << " trees into: " << modelOutDir << "\n";

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    multi.serialize(ss);
    ss.seekg(0);
    auto multi2 = modelbuilder::MultiModelBuilder::deserialize(ss);
    multi2.loadMap();

    auto treeArtifact = multi2.getTree(wantTarget);
    std::cout << "Fetched tree for targetColumn=" << wantTarget
              << " with elements=" << treeArtifact.tree().elementCount() << "\n";

    // Example prediction: all missing (0) -> conditional mixture.
    std::vector<std::uint64_t> sample(cols, 0);
    const auto dist = multi2.predict(sample, wantTarget, /*applyConditional=*/true);
    std::cout << "Prediction distribution (targetColumn=" << wantTarget << "):\n";
    for (const auto& [value, p] : dist) {
      std::cout << "  value=" << value << "  p=" << p << "\n";
    }

  } catch (const std::exception& e) {
    std::cout << "ModelBuilder_example_fast failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

