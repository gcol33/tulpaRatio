@echo off
REM Run Catch2 tests with coverage for ratiod C++ code
REM Execute from src/tests directory
REM Requires gcov (included with Rtools)

set PATH=C:\rtools45\x86_64-w64-mingw32.static.posix\bin;%PATH%

echo Cleaning old coverage files...
del /q *.gcno *.gcda *.gcov 2>nul

echo.
echo Compiling tests with coverage...
g++ -std=c++17 -Wall -Wextra -I.. -I. --coverage -fprofile-arcs -ftest-coverage -O0 -g -o run_tests_cov.exe test_main.cpp test_linalg.cpp test_zi.cpp

if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    exit /b 1
)

echo.
echo Running tests...
run_tests_cov.exe

echo.
echo === Coverage Summary ===
gcov -r test_linalg.cpp test_zi.cpp 2>nul | findstr /C:"File" /C:"Lines"

echo.
echo Coverage data files generated. Install lcov for HTML reports.
