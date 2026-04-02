#include <ModelBuilder/MultiModelBuilder.hpp>

#include <DataTable/DataTable.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>

#include "ModelBuilder/ModelBuilder.hpp"

int main(int argc, char** argv) {
  // This example mirrors ModelBuilder_example.cpp, except it does NOT build the model.
  // It assumes the model was already built previously into modelOutDir and simply opens it.
  //
  // Usage:
  //   ./ModelBuilder_example_build_serialized_model [--reparse]

  const std::filesystem::path csvPath = std::filesystem::path{"examples"} / "sample_data" / "ptsd.csv";
  const std::filesystem::path parsedDir = std::filesystem::path{"build"} / "datasets" / "ptsd_output";
  const std::filesystem::path modelOutDir = std::filesystem::path{"build"} / "models" / "ptsd_model";

  const bool forceReparse = (argc > 1 && std::string_view(argv[1]) == "--reparse");

  try {
    // Parse/load the dataset only to get column count for sizing the prediction vector.
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

    // Open a previously-built model artifact.
    auto multi = modelbuilder::MultiModelBuilder::open(modelOutDir.string());

    std::cout << "Opened " << multi.treeCount() << " trees from: " << modelOutDir << "\n";
    std::cout << "  - mb_trees.bin\n";
    std::cout << "  - mb_map.bin\n";

    // Fetch one TreeBuilder for a specific target.
    const std::uint64_t wantTarget = dt.getColumnIndex("PTSDDx");
    auto treeArtifact = multi.getTree(wantTarget);

    std::cout << "Fetched tree for targetColumn=" << wantTarget
              << " with elements=" << treeArtifact.tree().elementCount() << "\n";
    modelbuilder::ModelBuilder builder;
    builder.createGraphviz(treeArtifact.tree(), "col_1_tree.dot");
    // Example prediction: create a sample vector sized to the dataset column count.
    // Here we mark everything missing (0) and request conditional prediction.
    std::vector<std::uint64_t> sample(cols, 0);
    const auto dist = multi.predict(sample, wantTarget, /*applyConditional=*/true);
    std::cout << "Prediction distribution (targetColumn=" << wantTarget << "):\n";
    for (const auto& [value, p] : dist) {
      std::cout << "  value=" << dt.getColumnValue(wantTarget,value) << "  p=" << p << "\n";
    }

  } catch (const std::exception& e) {
    std::cout << "ModelBuilder_example_build_serialized_model failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
