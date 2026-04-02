add_executable(TreeBuilder_example
  TreeBuilder_example.cpp
)

target_link_libraries(TreeBuilder_example
  PRIVATE
    ModelBuilder::ModelBuilder
)

add_executable(ModelBuilder_example
  ModelBuilder_example.cpp
)

target_link_libraries(ModelBuilder_example
  PRIVATE
    ModelBuilder::ModelBuilder
)

add_executable(ModelBuilder_example_fast
  ModelBuilder_example_fast.cpp
)

target_link_libraries(ModelBuilder_example_fast
  PRIVATE
    ModelBuilder::ModelBuilder
)

