add_test([=[ModelBuilder.Add]=]  /Volumes/DockDrive/github/research/ModelBuilder/build/tests/modelbuilder_tests [==[--gtest_filter=ModelBuilder.Add]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ModelBuilder.Add]=]  PROPERTIES WORKING_DIRECTORY /Volumes/DockDrive/github/research/ModelBuilder/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  modelbuilder_tests_TESTS ModelBuilder.Add)
