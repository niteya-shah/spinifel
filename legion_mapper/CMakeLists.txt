cmake_minimum_required(VERSION 3.1)
project(legion_mappers)

find_package(Legion REQUIRED)

list(APPEND LEGION_MAPPERS_SRC
  lifeline_mapper.cc       lifeline_mapper.h
  simple_mapper.cc         simple_mapper.h
)

add_library(legion_mappers SHARED ${LEGION_MAPPERS_SRC})
target_link_libraries(legion_mappers Legion::Legion)

set_property(TARGET legion_mappers PROPERTY CXX_STANDARD 11)
set_property(TARGET legion_mappers PROPERTY CXX_STANDARD_REQUIRED ON)
