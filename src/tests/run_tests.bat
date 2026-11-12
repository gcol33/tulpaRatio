@echo off
REM Run Catch2 tests for ratiod C++ code
REM Execute from src/tests directory

set PATH=C:\rtools45\x86_64-w64-mingw32.static.posix\bin;%PATH%

echo Compiling tests...
g++ -std=c++17 -Wall -Wextra -I.. -I. -O2 -o run_tests.exe test_main.cpp test_linalg.cpp test_zi.cpp

if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    exit /b 1
)

echo.
echo Running tests...
run_tests.exe %*
