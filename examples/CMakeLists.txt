add_executable(modelbuilder_example
  example_main.cpp
)

target_link_libraries(modelbuilder_example
  PRIVATE
    ModelBuilder::ModelBuilder
)

