========================
OpenGL Maze Game
========================

A small 3D maze game written in C++17 + OpenGL 3.3 Core.
It generates a perfect maze procedurally and lets you explore it in first-person.

========================
✨ Features
========================
- Procedural maze generation (iterative DFS)
- First-person controls (WASD + mouse look, Shift = run, Space = jump)
- Collision detection (walls & bounds)
- Minimap (press M) showing entire maze, centered in window, with player marker
- ESC to return from minimap, toggle mouse capture
- Runtime maze size input (prompt or CLI args)
- Built with GLFW + GLAD + GLM via vcpkg

========================
🎮 Controls
========================
- W/A/S/D  → Move
- Mouse    → Look around
- Shift    → Run
- Space    → Jump
- M        → Show minimap
- ESC      → Back to game / toggle cursor

========================
🛠️ Build Instructions (Windows + vcpkg)
========================

1. Install vcpkg and bootstrap it:
   git clone https://github.com/microsoft/vcpkg
   .\vcpkg\bootstrap-vcpkg.bat

2. Install dependencies:
   vcpkg install glfw3 glm

3. Clone this repo:
   git clone https://github.com/yourusername/openGL-maze.git
   cd openGL-maze

4. Configure and build:
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
   cmake --build . --config Release

5. Run:
   .\Release\opengl_maze.exe

========================
📦 Dependencies
========================
- GLFW – window/context & input
- GLAD – OpenGL loader
- GLM  – math library
- CMake – build system

========================
🚀 Future Ideas
========================
- Textured walls and lighting effects
- Fog of war or path overlay on minimap
- Multi-floor mazes
- Web build via Emscripten

