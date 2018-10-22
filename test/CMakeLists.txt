add_subdirectory(python)

option(CHECK_PEP8 "add test to check whether code follows autopep8" ON)

execute_process(COMMAND autopep8 --version RESULT_VARIABLE HAS_AUTOPEP8 OUTPUT_QUIET ERROR_QUIET)
if("${HAS_AUTOPEP8}" STREQUAL "0" AND ${CHECK_PEP8})
    add_test(NAME autopep8 COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/check_autopep8.sh" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
else()
    message(STATUS "autopep8 not found or CHECK_PEP8 is OFF. Will not check if code follows PEP8.")
endif()
