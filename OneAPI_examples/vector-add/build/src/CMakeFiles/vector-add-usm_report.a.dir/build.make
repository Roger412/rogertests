# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build

# Include any dependencies generated for this target.
include src/CMakeFiles/vector-add-usm_report.a.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/vector-add-usm_report.a.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/vector-add-usm_report.a.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/vector-add-usm_report.a.dir/flags.make

src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o: src/CMakeFiles/vector-add-usm_report.a.dir/flags.make
src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o: ../src/vector-add-usm.cpp
src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o: src/CMakeFiles/vector-add-usm_report.a.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o"
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src && /opt/intel/oneapi/compiler/2025.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o -MF CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o.d -o CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o -c /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/src/vector-add-usm.cpp

src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.i"
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src && /opt/intel/oneapi/compiler/2025.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/src/vector-add-usm.cpp > CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.i

src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.s"
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src && /opt/intel/oneapi/compiler/2025.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/src/vector-add-usm.cpp -o CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.s

# Object files for target vector-add-usm_report.a
vector__add__usm_report_a_OBJECTS = \
"CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o"

# External object files for target vector-add-usm_report.a
vector__add__usm_report_a_EXTERNAL_OBJECTS =

vector-add-usm_report.a: src/CMakeFiles/vector-add-usm_report.a.dir/vector-add-usm.cpp.o
vector-add-usm_report.a: src/CMakeFiles/vector-add-usm_report.a.dir/build.make
vector-add-usm_report.a: src/CMakeFiles/vector-add-usm_report.a.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../vector-add-usm_report.a"
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector-add-usm_report.a.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/vector-add-usm_report.a.dir/build: vector-add-usm_report.a
.PHONY : src/CMakeFiles/vector-add-usm_report.a.dir/build

src/CMakeFiles/vector-add-usm_report.a.dir/clean:
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src && $(CMAKE_COMMAND) -P CMakeFiles/vector-add-usm_report.a.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/vector-add-usm_report.a.dir/clean

src/CMakeFiles/vector-add-usm_report.a.dir/depend:
	cd /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/src /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src /home/roger/Github/rogertests/rogertests/OneAPI_examples/vector-add/build/src/CMakeFiles/vector-add-usm_report.a.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/vector-add-usm_report.a.dir/depend

