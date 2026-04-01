#include <ModelBuilder/ModelBuilder.hpp>
#include <ModelBuilder/TreeBuilder.hpp>

#include <DataTable/DataTable.h>

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  // ModelBuilder remains available as an interactive builder / utilities host.
  // TreeBuilder is the serializable single-tree artifact.
  modelbuilder::ModelBuilder mb;

  // Input CSV (categorical data expected).
  const std::filesystem::path csvPath = std::filesystem::path{"examples"} / "sample_data" / "ptsd.csv";

  // DataTable parses CSVs into an output directory layout (meta_data/ + mapped_data/).
  // That parsed directory is what FeatureSelector/ModelBuilder expects to load.
  const std::filesystem::path parsedDir = std::filesystem::path{"build"} / "datasets" / "ptsd_output";

  const bool forceReparse = (argc > 1 && std::string_view(argv[1]) == "--reparse");

  try {
    DataTableLib::DataTable dt;
    const auto metaDir = parsedDir / "meta_data";
    if (!forceReparse && std::filesystem::exists(metaDir)) {
      // Fast path: reuse previous parse output.
      dt.load(parsedDir.string());
      std::cout << "Loaded parsed dataset: " << parsedDir << "\n";
    } else {
      // Parse CSV -> output directory (safe to re-run; it will overwrite/update outputs).
      dt.setInputFilePath(csvPath.string());
      dt.setOutputDirectory(parsedDir.string());
      dt.parse(/*threads=*/1);
      std::cout << "Parsed CSV: " << csvPath << " -> " << parsedDir << "\n";
    }
    std::cout << "Columns: " << dt.getColumnCount() << ", Rows (incl header): " << dt.getRowCount() << "\n";
    if (dt.getColumnCount() > 0) {
      std::cout << "First column header: " << dt.getColumnHeader(0) << "\n";
    }

    // Configure build parameters.
    // Column 0 is an identifier (record_id) and must be excluded from both target and split candidates.
    const std::size_t targetColumn = 1;
    const std::size_t maxDepth = 0; // 0 => unlimited

    // Significance thresholds.
    const double columnAlpha = 0.05;
    const bool columnAlphaBonferroni = true;
    const double partitionAlpha = 0.05;
    const bool partitionAlphaBonferroni = false;

    // Optional: restrict split candidate columns for building/training.
    // These analysis columns are NOT stored in the resulting TreeBuilder artifact serialization.
    // For a simple example, enable all columns except id and target.
    std::vector<std::size_t> analysisColumns;
    analysisColumns.reserve(static_cast<std::size_t>(dt.getColumnCount()));
    for (std::size_t c = 0; c < static_cast<std::size_t>(dt.getColumnCount()); ++c) {
      if (c == 0) continue; // exclude id column
      if (c == targetColumn) continue; // exclude target column
      analysisColumns.push_back(c);
    }

    // Build the serializable artifact.
    auto artifact = modelbuilder::TreeBuilder::buildFromDataDir(parsedDir.string(),
                                                                targetColumn,
                                                                maxDepth,
                                                                columnAlpha,
                                                                columnAlphaBonferroni,
                                                                partitionAlpha,
                                                                partitionAlphaBonferroni,
                                                                analysisColumns);

    const auto& tree = artifact.tree();
    std::cout << "Built tree with " << tree.elementCount() << " elements\n";
    mb.createGraphviz(tree, "modelbuilder_tree.dot");
    std::cout << "Wrote modelbuilder_tree.dot (render with: dot -Tpng modelbuilder_tree.dot -o tree.png)\n";
  } catch (const std::exception& e) {
    std::cout << "ModelBuilder example failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

