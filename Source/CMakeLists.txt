add_executable(fastwindterrain
    Grid.cpp
    Terrain.cpp
    Output.cpp
    main.cpp
)

target_include_directories(fastwindterrain
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fastwindterrain PRIVATE AMReX::amrex_3d)
