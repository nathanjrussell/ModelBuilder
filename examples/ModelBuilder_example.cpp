#include <ModelBuilder/MultiModelBuilder.hpp>

#include <DataTable/DataTable.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  // This example demonstrates the multi-tree artifact:
  //  - build many target-column trees in parallel
  //  - write mb_trees.bin + mb_map.bin
  //  - serialize/deserialize the MultiModelBuilder object itself
  //  - getTree(targetColumn) to lazily fetch one TreeBuilder

  const std::filesystem::path csvPath = std::filesystem::path{"examples"} / "sample_data" / "ptsd.csv";
  const std::filesystem::path parsedDir = std::filesystem::path{"build"} / "datasets" / "ptsd_output";
  const std::filesystem::path modelOutDir = std::filesystem::path{"build"} / "models" / "ptsd_model";

  const bool forceReparse = (argc > 1 && std::string_view(argv[1]) == "--reparse");

  try {
    // Parse once using DataTable.
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
    if (cols < 3) {
      throw std::runtime_error("Need at least 3 columns for this example (id + >=2 features)");
    }

    // Parameters.
    const std::size_t maxDepth = 0; // unlimited
    const double columnAlpha = 0.005;
    const bool columnAlphaBonferroni = true;
    const double partitionAlpha = 0.005;
    const bool partitionAlphaBonferroni = false;

    // analysisColumns: all columns except column 0 (id).
    std::vector<std::size_t> analysisColumns;
    analysisColumns.reserve(cols);
    for (std::size_t c = 0; c < cols; ++c) {
      if (c == 0) continue;
      analysisColumns.push_back(c);
    }

    // targetColumns: all columns except column 0 (id).
    // NOTE: per-tree build will remove the target column from analysis columns.
    std::vector<std::size_t> targetColumns = analysisColumns;

    const std::size_t threadCount = 16;

    // Build and write out the multi-tree artifact.
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
    std::cout << "  - mb_trees.bin\n";
    std::cout << "  - mb_map.bin\n";

    // Serialize/deserialize the MultiModelBuilder object itself (metadata + cached map).
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    // multi.serialize(ss);
    ss.seekg(0);
    auto multi2 = modelbuilder::MultiModelBuilder::deserialize(ss);

    // Ensure lazy map is available even if we "forgot" it.
    // (deserialize() already loads an embedded copy, but loadMap() will rebuild from disk.)
    multi2.loadMap();

    // Fetch one TreeBuilder for a specific target.
    const std::uint64_t wantTarget = 1;
    auto treeArtifact = multi2.getTree(wantTarget);

    std::cout << "Fetched tree for targetColumn=" << wantTarget
              << " with elements=" << treeArtifact.tree().elementCount() << "\n";

    // Example prediction: create a sample vector sized to the dataset column count.
    // Here we mark everything missing (0) and request conditional prediction.
    std::vector<std::uint64_t> sample(cols, 0);
    const auto dist = multi2.predict(sample, wantTarget, /*applyConditional=*/true);
    std::cout << "Prediction distribution (targetColumn=" << wantTarget << "):\n";
    for (const auto& [value, p] : dist) {
      std::cout << "  value=" << value << "  p=" << p << "\n";
    }

  } catch (const std::exception& e) {
    std::cout << "ModelBuilder_example failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}


