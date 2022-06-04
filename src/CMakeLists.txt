find_package (Python3 REQUIRED COMPONENTS Interpreter)


add_custom_target(gen-dataset
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/data-extractor.sh ${JULIET_TEST_SUITE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/data-extractor.py ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/dataset/
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating dataset")
