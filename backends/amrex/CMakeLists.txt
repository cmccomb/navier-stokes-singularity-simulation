cmake_minimum_required(VERSION 3.24)
project(ns_static_refinement LANGUAGES C CXX)
include(FetchContent)

# Opt in without changing the archived serial baseline or numerical precision.
option(NS_ENABLE_OPENMP "Enable bounded CPU threading in AMReX and Hydro" OFF)

# Immutable upstream source snapshots, with archive integrity checks.
set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
set(AMReX_PRECISION DOUBLE CACHE STRING "" FORCE)
set(AMReX_MPI OFF CACHE BOOL "" FORCE)
set(AMReX_OMP ${NS_ENABLE_OPENMP} CACHE BOOL "" FORCE)
set(AMReX_FORTRAN OFF CACHE BOOL "" FORCE)
set(AMReX_PARTICLES OFF CACHE BOOL "" FORCE)
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_EB OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_EM OFF CACHE BOOL "" FORCE)
set(AMReX_GPU_BACKEND NONE CACHE STRING "" FORCE)
FetchContent_Declare(amrex
  URL https://codeload.github.com/AMReX-Codes/amrex/tar.gz/e60cdc18711ccf7fc0616d7a2fdf062021376976
  URL_HASH SHA256=8d8aa5990ad0f76326108084afcd2180cf9da431dc37fa99ad631318cd6a5fae)
FetchContent_MakeAvailable(amrex)

set(HYDRO_SPACEDIM 3 CACHE STRING "" FORCE)
set(HYDRO_MPI OFF CACHE BOOL "" FORCE)
set(HYDRO_OMP ${NS_ENABLE_OPENMP} CACHE BOOL "" FORCE)
set(HYDRO_EB OFF CACHE BOOL "" FORCE)
set(HYDRO_NO_EB ON CACHE BOOL "" FORCE)
set(HYDRO_PROJECTIONS ON CACHE BOOL "" FORCE)
FetchContent_Declare(hydro
  URL https://codeload.github.com/AMReX-Fluids/AMReX-Hydro/tar.gz/e49df248aabd2cc11865eb5be734a2f5f2f65ee5
  URL_HASH SHA256=00d847bc993508c80a42e746396d1401039360ed67d5201dc8a3d17225855a09)
FetchContent_MakeAvailable(hydro)

add_executable(ns_projection_pilot projection_pilot.cpp)
target_compile_features(ns_projection_pilot PRIVATE cxx_std_20)
target_link_libraries(ns_projection_pilot PRIVATE amrex_hydro)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/projection_pilot.cpp" pilot_source_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS projection_pilot.cpp)
target_compile_definitions(ns_projection_pilot PRIVATE
  NS_PILOT_SOURCE_SHA256="${pilot_source_sha256}"
  NS_AMREX_COMMIT="e60cdc18711ccf7fc0616d7a2fdf062021376976"
  NS_HYDRO_COMMIT="e49df248aabd2cc11865eb5be734a2f5f2f65ee5")
enable_testing()
add_executable(ns_threading_check threading_check.cpp)
target_compile_features(ns_threading_check PRIVATE cxx_std_20)
target_link_libraries(ns_threading_check PRIVATE amrex_hydro)
add_test(NAME forcing_serial_blocks COMMAND ns_threading_check ns.force=shear ns.force_threads=1)
if(NS_ENABLE_OPENMP)
  add_test(NAME forcing_threaded_blocks COMMAND ns_threading_check ns.force=shear ns.force_threads=2)
  add_test(NAME forcing_thread_cap COMMAND ns_threading_check ns.force=shear ns.force_threads=4)
endif()

add_executable(ns_profile_probe profile_probe.cpp)
target_compile_features(ns_profile_probe PRIVATE cxx_std_20)
add_executable(ns_force_probe force_probe.cpp)
target_compile_features(ns_force_probe PRIVATE cxx_std_20)
target_link_libraries(ns_force_probe PRIVATE amrex_hydro)
add_test(NAME exact_rest COMMAND ns_projection_pilot n_cell=16 levels=3 test_case=rest)
add_test(NAME mixed_projection COMMAND ns_projection_pilot n_cell=16 levels=3 test_case=mixed)
add_test(NAME solenoidal COMMAND ns_projection_pilot n_cell=16 levels=4 test_case=solenoidal)
add_test(NAME four_level_projection COMMAND ns_projection_pilot n_cell=16 levels=4 test_case=mixed)

add_executable(ns_diffusion_pilot diffusion_pilot.cpp)
target_compile_features(ns_diffusion_pilot PRIVATE cxx_std_20)
target_link_libraries(ns_diffusion_pilot PRIVATE amrex_hydro)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/diffusion_pilot.cpp" diffusion_source_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS diffusion_pilot.cpp)
target_compile_definitions(ns_diffusion_pilot PRIVATE
  NS_DIFFUSION_SOURCE_SHA256="${diffusion_source_sha256}"
  NS_AMREX_COMMIT="e60cdc18711ccf7fc0616d7a2fdf062021376976")
add_test(NAME diffusion_rest COMMAND ns_diffusion_pilot test_case=rest steps=8)
add_test(NAME diffusion_constant COMMAND ns_diffusion_pilot test_case=constant steps=8)
add_test(NAME diffusion_decay COMMAND ns_diffusion_pilot test_case=decay steps=8)

# Build the established coupled integrator against the same pinned operators.
# Include its src directory, not its superbuild, to avoid duplicate libraries.
option(NS_BUILD_INCFLO "Build the upstream coupled incompressible integrator" ON)
if(NS_BUILD_INCFLO)
  FetchContent_Declare(incflo_upstream
    URL https://codeload.github.com/AMReX-Fluids/incflo/tar.gz/7491eea4d69bbfde0581a5fe7f95d803841a8a09
    URL_HASH SHA256=51ef0cbc023ae98d1be559bff6c3a76e1bf951e3576f65a5c9308670148efc0d
    SOURCE_SUBDIR no-superbuild)
  FetchContent_MakeAvailable(incflo_upstream)
  set(INCFLO_DIM 3)
  set(INCFLO_EB OFF)
  set(INCFLO_PARTICLES OFF)
  add_executable(incflo)
  set_target_properties(incflo PROPERTIES OUTPUT_NAME ns_incflo)
  target_compile_features(incflo PRIVATE cxx_std_20)
  target_link_libraries(incflo PRIVATE amrex_hydro)
  list(APPEND CMAKE_MODULE_PATH "${amrex_SOURCE_DIR}/Tools/CMake")
  include(AMReXBuildInfo)
  generate_buildinfo(incflo ${CMAKE_CURRENT_SOURCE_DIR})
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS incflo_overlay.py ns_case.H paper_profile.H paper_fields.H)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/ns_case.H" ns_case_hash)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/incflo_overlay.py" ns_overlay_hash)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/paper_profile.H" ns_profile_hash)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/paper_fields.H" ns_fields_hash)
  string(SHA256 ns_adapter_hash "${ns_case_hash};${ns_overlay_hash};${ns_profile_hash};${ns_fields_hash}")
  target_compile_definitions(incflo PRIVATE NS_ADAPTER_SHA256="${ns_adapter_hash}")
  execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/incflo_overlay.py"
    "${incflo_upstream_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/incflo-overlay"
    COMMAND_ERROR_IS_FATAL ANY)
  target_include_directories(incflo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/incflo-overlay" incflo-src)
  add_test(NAME coupled_rest COMMAND incflo "${CMAKE_CURRENT_SOURCE_DIR}/inputs.coupled"
    stop_time=0.025 amr.max_level=2 "ns.refine_half_width=0.5 0.25")
  set_tests_properties(coupled_rest PROPERTIES TIMEOUT 60)
  add_test(NAME coupled_nonbinary_volume COMMAND incflo "${CMAKE_CURRENT_SOURCE_DIR}/inputs.coupled"
    stop_time=0.003125 ns.max_dt=0.003125 "amr.n_cell=96 96 96"
    amr.max_level=3 "ns.refine_half_width=0.5 0.25 0.125")
  set_tests_properties(coupled_nonbinary_volume PROPERTIES TIMEOUT 180)
  add_executable(ns_archive_check archive_check.cpp)
  target_compile_features(ns_archive_check PRIVATE cxx_std_20)
  target_link_libraries(ns_archive_check PRIVATE amrex_hydro)
endif()

# Bound regression resource use even when the developer's shell has no OMP cap.
if(NS_ENABLE_OPENMP)
  get_property(ns_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${ns_tests} PROPERTIES PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_DYNAMIC=FALSE;OMP_THREAD_LIMIT=2")
endif()
