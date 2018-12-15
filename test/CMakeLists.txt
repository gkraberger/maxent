# Copy h5-files to binary dir
macro(copyfiles)
    FILE(GLOB all_h5_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h5)
    FILE(GLOB all_dat_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.dat)
    FILE(GLOB all_npz_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.npz)
    FILE(GLOB all_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_dat_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_ref_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_npz_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/matplotlibrc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endmacro()

macro(addtests)
    if(${TRIQS_V1})
      find_package(TriqsTest)

      if(${TEST_COVERAGE})
        set(PythonBuildExecutable ${PYTRIQS_COVERAGE_EXECUTABLE})
        include(${CMAKE_SOURCE_DIR}/cmake/triqs_add_python_test_coverage.cmake)
      endif()

      foreach(t ${all_tests})
        triqs_add_python_test(${t})
      endforeach()
    else() #TRIQS 2.0 and USE_TRIQS=OFF
      set(python_executable python)

      if(${TEST_COVERAGE})
          set(python_executable ${PYTHON_COVERAGE} run --append --source "${CMAKE_BINARY_DIR}/" )
      endif()

      foreach(t ${all_tests})
        add_test(NAME ${t} COMMAND ${python_executable} ${CMAKE_CURRENT_SOURCE_DIR}/${t}.py)
      endforeach()

      set_property(TEST ${all_tests} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH} )
    endif()
endmacro()

add_subdirectory(general)

if(${USE_TRIQS})
    add_subdirectory(triqs)
endif()
