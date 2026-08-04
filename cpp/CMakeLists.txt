cmake_minimum_required(VERSION 3.20)

project(ddioc_trajectory_planning_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(ddioc_synthetic_ioc
    src/dynamics_learning.cpp
    src/hlo_learning.cpp
    src/linear_ioc_bilevel.cpp
    src/pipeline.cpp
    src/synthetic_ioc.cpp
)

target_include_directories(ddioc_synthetic_ioc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ddioc_synthetic_ioc PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(ddioc_synthetic_ioc PRIVATE /W4)
else()
    target_compile_options(ddioc_synthetic_ioc PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(ddioc_synthetic_demo
    src/main.cpp
)

target_link_libraries(ddioc_synthetic_demo PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_lioc_benchmark
    src/lioc_main.cpp
)

target_link_libraries(ddioc_lioc_benchmark PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_validate_dynamics
    src/validate_dynamics_main.cpp
)

target_link_libraries(ddioc_validate_dynamics PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_pipeline
    src/pipeline_main.cpp
)

target_link_libraries(ddioc_pipeline PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_smoke_test
    tests/smoke_test.cpp
)

target_link_libraries(ddioc_smoke_test PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_lioc_smoke_test
    tests/lioc_smoke_test.cpp
)

target_link_libraries(ddioc_lioc_smoke_test PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_dynamics_smoke_test
    tests/dynamics_smoke_test.cpp
)

target_link_libraries(ddioc_dynamics_smoke_test PRIVATE ddioc_synthetic_ioc)

add_executable(ddioc_pipeline_smoke_test
    tests/pipeline_smoke_test.cpp
)

target_link_libraries(ddioc_pipeline_smoke_test PRIVATE ddioc_synthetic_ioc)

if(BUILD_TESTING)
    add_test(NAME ddioc_smoke_test COMMAND ddioc_smoke_test)
    add_test(NAME ddioc_lioc_smoke_test COMMAND ddioc_lioc_smoke_test)
    add_test(NAME ddioc_dynamics_smoke_test COMMAND ddioc_dynamics_smoke_test)
    add_test(NAME ddioc_pipeline_smoke_test COMMAND ddioc_pipeline_smoke_test)
endif()