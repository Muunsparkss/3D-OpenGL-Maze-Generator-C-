// main.cpp - OpenGL 3D Maze (GLFW + GLAD 1 + GLM)
// Features: fixed yaw, centered minimap (M), ESC to return, optional runtime maze size input

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>    // shuffle, clamp
#include <optional>
#include <random>
#include <stdexcept>
#include <cmath>

#include "glad.h"          // GLAD 1 header (keep BEFORE GLFW)
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

// -------------------- Maze generation (perfect maze via iterative DFS) --------------------
static const int DX[4] = {0, 1, 0, -1};
static const int DY[4] = {-1, 0, 1, 0};
enum Dir {N=0, E=1, S=2, W=3};

struct Cell { uint8_t open = 0; bool visited = false; };

struct Maze {
    int W, H;
    std::vector<Cell> g;
    inline int idx(int x,int y) const { return y*W + x; }
    inline bool inb(int x,int y) const { return x>=0 && x<W && y>=0 && y<H; }

    Maze(int w,int h, std::optional<unsigned> seed = {}) : W(w), H(h), g(W*H) {
        std::mt19937 rng;
        if (seed.has_value()) rng.seed(*seed); else rng.seed(std::random_device{}());
        struct Frame{int x,y; std::vector<int> dirs;};
        std::vector<Frame> st;
        g[idx(0,0)].visited = true;
        std::vector<int> d0 = {0,1,2,3}; std::shuffle(d0.begin(), d0.end(), rng);
        st.push_back({0,0,d0});
        while(!st.empty()){
            auto &f=st.back();
            if(f.dirs.empty()){ st.pop_back(); continue; }
            int d = f.dirs.back(); f.dirs.pop_back();
            int nx=f.x+DX[d], ny=f.y+DY[d];
            if(!inb(nx,ny) || g[idx(nx,ny)].visited) continue;
            g[idx(f.x,f.y)].open |= (1u<<d);
            g[idx(nx,ny)].open |= (1u<<((d+2)&3));
            std::vector<int> nd={0,1,2,3}; std::shuffle(nd.begin(), nd.end(), rng);
            g[idx(nx,ny)].visited=true; st.push_back({nx,ny,nd});
        }
    }
    bool isOpen(int x,int y,int d) const {
        if(!inb(x,y)) return false;
        return (g[idx(x,y)].open & (1u<<d)) != 0;
    }
};

// -------------------- Shader helpers --------------------
static const char* VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vNormal;
void main(){
    vNormal = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

static const char* FS = R"(#version 330 core
in vec3 vNormal;
out vec4 FragColor;
uniform vec3 uColor;
uniform vec3 uLightDir; // normalized
void main(){
    float ambient = 0.25;
    float diff = max(dot(normalize(vNormal), -uLightDir), 0.0);
    vec3 color = uColor * (ambient + 0.8*diff);
    FragColor = vec4(color, 1.0);
}
)";

static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){ char log[2048]; glGetShaderInfoLog(s,2048,nullptr,log); throw std::runtime_error(log); }
    return s;
}
static GLuint link(GLuint vs, GLuint fs){
    GLuint p=glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); throw std::runtime_error(log); }
    return p;
}

// -------------------- Geometry: unit cube (per-face normals) --------------------
struct Mesh {
    GLuint vao=0, vbo=0;
    GLsizei vertexCount=0;
    void destroy(){ if(vbo) glDeleteBuffers(1,&vbo); if(vao) glDeleteVertexArrays(1,&vao); }
};

static Mesh makeCube(){
    struct V{ float x,y,z, nx,ny,nz; };
    std::vector<V> v;
    auto addFace=[&](glm::vec3 n, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d){
        V A{a.x,a.y,a.z,n.x,n.y,n.z};
        V B{b.x,b.y,b.z,n.x,n.y,n.z};
        V C{c.x,c.y,c.z,n.x,n.y,n.z};
        V D{d.x,d.y,d.z,n.x,n.y,n.z};
        v.insert(v.end(), {A,B,C, A,C,D});
    };
    glm::vec3 p[8] = {
        {-0.5f,-0.5f,-0.5f},{0.5f,-0.5f,-0.5f},{0.5f,0.5f,-0.5f},{-0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f, 0.5f},{0.5f,-0.5f, 0.5f},{0.5f,0.5f, 0.5f},{-0.5f,0.5f, 0.5f}
    };
    addFace({0,0,-1}, p[0],p[1],p[2],p[3]);
    addFace({0,0, 1}, p[5],p[4],p[7],p[6]);
    addFace({-1,0,0}, p[4],p[0],p[3],p[7]);
    addFace({ 1,0,0}, p[1],p[5],p[6],p[2]);
    addFace({0,1,0},  p[3],p[2],p[6],p[7]);
    addFace({0,-1,0}, p[4],p[5],p[1],p[0]);

    Mesh m;
    glGenVertexArrays(1,&m.vao);
    glBindVertexArray(m.vao);
    glGenBuffers(1,&m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size()*sizeof(V)), v.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)(3*sizeof(float)));
    m.vertexCount = (GLsizei)v.size();
    glBindVertexArray(0);
    return m;
}

// -------------------- Camera --------------------
struct Camera {
    glm::vec3 pos{0,1.0f,0};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 3.0f; // Base walking speed
    float sens  = 0.12f;
    glm::vec3 forward() const {
        float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw));
        float cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch));
        return glm::normalize(glm::vec3(cy*cp, sp, -sy*cp));
    }
    glm::vec3 right() const { return glm::normalize(glm::cross(forward(), {0,1,0})); }
    glm::mat4 view() const { return glm::lookAt(pos, pos+forward(), {0,1,0}); }
};

// -------------------- Collision --------------------
struct MazeCollision {
    const Maze& mz;
    float cellSize = 2.0f;
    float radius   = 0.25f;
    MazeCollision(const Maze& m): mz(m) {}
    void move(glm::vec3& pos, const glm::vec3& vel, float dt){
        glm::vec3 np = pos + vel * dt;
        auto toCell = [&](float x){ return int(std::floor(x / cellSize)); };
        int cx = toCell(pos.x), cz = toCell(pos.z);
        int nx = toCell(np.x),  nz = toCell(np.z);
        if (nx != cx) {
            if (nx > cx) { if (!mz.isOpen(cx, cz, E)) np.x = (cx+1)*cellSize - radius*0.9f; }
            else {           if (!mz.isOpen(cx, cz, W)) np.x = (cx)*cellSize + radius*0.9f; }
        }
        if (nz != cz) {
            if (nz > cz) { if (!mz.isOpen(cx, cz, S)) np.z = (cz+1)*cellSize - radius*0.9f; }
            else {           if (!mz.isOpen(cx, cz, N)) np.z = (cz)*cellSize + radius*0.9f; }
        }
        np.x = std::clamp(np.x, 0.0f+radius, mz.W*cellSize - radius);
        np.z = std::clamp(np.z, 0.0f+radius, mz.H*cellSize - radius);
        pos = np;
    }
};

// -------------------- Build instances --------------------
struct Instance { glm::mat4 model; glm::vec3 color; };

static std::vector<Instance> buildMazeInstances(const Maze& m, float cell=2.0f, float wallH=2.2f, float wallT=0.1f){
    std::vector<Instance> inst;
    auto addBox = [&](float x,float z, float sx,float sy,float sz, glm::vec3 color){
        glm::mat4 M(1.0f);
        M = glm::translate(M, {x, sy*0.5f, z});
        M = glm::scale(M, {sx, sy, sz});
        inst.push_back({M, color});
    };

    glm::vec3 wallColor(0.4f,0.2f,0.8f);
    for(int x=0;x<m.W;x++){
        addBox(x*cell + cell*0.5f, 0.0f,      cell, wallH, wallT, wallColor);
        addBox(x*cell + cell*0.5f, m.H*cell,    cell, wallH, wallT, wallColor);
    }
    for(int z=0; z<m.H; z++){
        addBox(0.0f,       z*cell + cell*0.5f, wallT, wallH, cell, wallColor);
        addBox(m.W*cell,     z*cell + cell*0.5f, wallT, wallH, cell, wallColor);
    }
    for(int y=0;y<m.H;y++){
        for(int x=0;x<m.W;x++){
            float cx = x*cell + cell*0.5f;
            float cz = y*cell + cell*0.5f;
            if (!m.isOpen(x,y,N)) addBox(cx, y*cell,        cell, wallH, wallT, wallColor);
            if (!m.isOpen(x,y,S)) addBox(cx, (y+1)*cell,      cell, wallH, wallT, wallColor);
            if (!m.isOpen(x,y,W)) addBox(x*cell,         cz,  wallT, wallH, cell, wallColor);
            if (!m.isOpen(x,y,E)) addBox((x+1)*cell,     cz,  wallT, wallH, cell, wallColor);
        }
    }
    { // Floor
        glm::mat4 M(1.0f);
        M = glm::translate(M, {m.W*cell*0.5f, -0.01f, m.H*cell*0.5f});
        M = glm::scale(M, {m.W*cell, 0.02f, m.H*cell});
        inst.push_back({M, {0.25f,0.25f,0.28f}});
    }
    { // Start (green)
        float x = 0*cell + cell*0.5f;
        float z = 0*cell + cell*0.5f;
        glm::mat4 M(1.0f);
        M = glm::translate(M, {x, 0.01f, z});
        M = glm::scale(M, {cell*0.8f, 0.02f, cell*0.8f});
        inst.push_back({M, {0.1f,0.8f,0.2f}});
    }
    { // Exit (red)
        float x = (m.W-1)*cell + cell*0.5f;
        float z = (m.H-1)*cell + cell*0.5f;
        glm::mat4 M(1.0f);
        M = glm::translate(M, {x, 0.01f, z});
        M = glm::scale(M, {cell*0.8f, 0.02f, cell*0.8f});
        inst.push_back({M, {0.85f,0.2f,0.2f}});
    }
    return inst;
}

// -------------------- Globals --------------------
static bool mouseCaptured = true;
static double lastX=0, lastY=0;

// Minimap state & edge-triggered keys
static bool minimap = false;
static int  prevMState     = GLFW_RELEASE;
static int  prevEscState = GLFW_RELEASE;

// >>> JUMPING, PHYSICS, AND MOVEMENT VARIABLES (Added RUN_SPEED_MULTIPLIER) <<<
static float yVelocity = 0.0f;
static const float GRAVITY = 9.81f;        // Gravity (world units/sec^2)
static const float JUMP_STRENGTH = 10.0f;   // Initial upward speed for a jump
static const float PLAYER_EYE_HEIGHT = 1.0f; // The camera's fixed height when standing on the floor
static const float RUN_SPEED_MULTIPLIER = 2.0f; // Run twice as fast as base speed
static bool isGrounded = true;             // Flag to prevent mid-air jumping
// ------------------------------------------

int main(int argc, char** argv){
    // ---- Maze size: args OR interactive input ----
    int W = 21, H = 11;
    if (argc>=3) {
        W = std::max(2, std::atoi(argv[1]));
        H = std::max(2, std::atoi(argv[2]));
    } else {
        std::cout << "Enter maze width height (>=2), e.g. 35 21: ";
        int inW=0,inH=0;
        if (std::cin >> inW >> inH) {
            W = std::max(2, inW);
            H = std::max(2, inH);
        }
    }
    std::optional<unsigned> seed; if(argc>=4) seed = (unsigned)std::strtoul(argv[3],nullptr,10);

    Maze mz(W,H,seed);

    if(!glfwInit()){ std::cerr<<"GLFW init failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720, "OpenGL Maze (WASD+Mouse) | Press M for map", nullptr, nullptr);
    if(!win){ std::cerr<<"Window creation failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ std::cerr<<"GLAD load failed\n"; return 1; }

    auto setTitle = [&](const char* t){ glfwSetWindowTitle(win, t); };

    Camera cam;
    float cell = 2.0f;
    cam.pos = {cell*0.5f, PLAYER_EYE_HEIGHT, cell*0.7f}; // Use PLAYER_EYE_HEIGHT

    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetCursorPos(win, &lastX, &lastY);
    glfwSetWindowUserPointer(win, &cam);
    glfwSetCursorPosCallback(win, [](GLFWwindow* w, double xpos, double ypos){
        if(!mouseCaptured) return;
        static double lx=lastX, ly=lastY;
        double dx = xpos - lx;
        double dy = ypos - ly;
        lx = xpos; ly = ypos;
        auto* c = (Camera*)glfwGetWindowUserPointer(w);
        if(!c) return;
        c->yaw  -= (float)dx * c->sens;    // right mouse -> turn right
        c->pitch -= (float)dy * c->sens;
        c->pitch = std::clamp(c->pitch, -89.0f, 89.0f);
        lastX = xpos; lastY = ypos;
    });

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    GLuint vs=compile(GL_VERTEX_SHADER, VS);
    GLuint fs=compile(GL_FRAGMENT_SHADER, FS);
    GLuint prog=link(vs,fs);
    glDeleteShader(vs); glDeleteShader(fs);

    Mesh cube = makeCube();
    auto instances = buildMazeInstances(mz, cell, 2.2f, 0.1f);

    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uModel = glGetUniformLocation(prog, "uModel");
    GLint uNormalMat = glGetUniformLocation(prog, "uNormalMat");
    GLint uColor = glGetUniformLocation(prog, "uColor");
    GLint uLightDir = glGetUniformLocation(prog, "uLightDir");

    glfwSetFramebufferSizeCallback(win, [](GLFWwindow*, int, int){ /* viewport set per-frame */ });

    MazeCollision collider(mz);
    double lastTime = glfwGetTime();

    while(!glfwWindowShouldClose(win)){
        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;

        // ---- Edge-triggered key handling ----
        int mState   = glfwGetKey(win, GLFW_KEY_M);
        int escState = glfwGetKey(win, GLFW_KEY_ESCAPE);

        if (mState == GLFW_PRESS && prevMState == GLFW_RELEASE) {
            minimap = true;
            mouseCaptured = false;
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            setTitle("Minimap - Press ESC to return");
        }
        prevMState = mState;

        if (escState == GLFW_PRESS && prevEscState == GLFW_RELEASE) {
            if (minimap) {
                minimap = false;
                mouseCaptured = true;
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                setTitle("OpenGL Maze (WASD+Mouse) | Press M for map");
            } else {
                mouseCaptured = !mouseCaptured;
                glfwSetInputMode(win, GLFW_CURSOR, mouseCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            }
        }
        prevEscState = escState;

        // ---- Movement and Physics Update (REPLACING OLD BLOCK) ----
        auto F = cam.forward();
        auto R = cam.right();
        glm::vec3 horizontalMove(0.0f); // Stores desired movement direction (X/Z only)
        
        if (!minimap) {
            // 1. Calculate desired horizontal movement vector (X-Z plane)
            if(glfwGetKey(win, GLFW_KEY_W)==GLFW_PRESS) horizontalMove += glm::normalize(glm::vec3(F.x,0,F.z));
            if(glfwGetKey(win, GLFW_KEY_S)==GLFW_PRESS) horizontalMove -= glm::normalize(glm::vec3(F.x,0,F.z));
            if(glfwGetKey(win, GLFW_KEY_A)==GLFW_PRESS) horizontalMove -= glm::normalize(glm::vec3(R.x,0,R.z));
            if(glfwGetKey(win, GLFW_KEY_D)==GLFW_PRESS) horizontalMove += glm::normalize(glm::vec3(R.x,0,R.z));

            // Determine current speed based on SHIFT key
            float currentSpeed = cam.speed;
            if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
                glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                currentSpeed *= RUN_SPEED_MULTIPLIER;
            }

            // Normalize horizontal movement and apply speed
            if (glm::length(horizontalMove) > 0.0f) {
                horizontalMove = glm::normalize(horizontalMove) * currentSpeed; // Use currentSpeed here
            }

            // 2. Jumping Input (only if grounded)
            if(glfwGetKey(win, GLFW_KEY_SPACE)==GLFW_PRESS && isGrounded) {
                yVelocity = JUMP_STRENGTH;
                isGrounded = false;
            }
        }
        
        // 3. Apply Gravity to Vertical Velocity
        if (!isGrounded) {
            yVelocity -= GRAVITY * dt;
        }

        // 4. Horizontal Collision/Movement (Uses existing MazeCollision function)
        // The collider only cares about X and Z
        glm::vec3 horizontalVel = {horizontalMove.x, 0.0f, horizontalMove.z};
        collider.move(cam.pos, horizontalVel, dt);

        // 5. Vertical Movement Application and Ground Clamping
        cam.pos.y += yVelocity * dt;

        if (cam.pos.y <= PLAYER_EYE_HEIGHT) {
            cam.pos.y = PLAYER_EYE_HEIGHT; // Snap to ground
            yVelocity = 0.0f;             // Stop falling/bouncing
            isGrounded = true;            // Mark as grounded
        }
        // ------------------------------------------

        // ---- Clear whole window ----
        glClearColor(0.05f,0.06f,0.08f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // Window *framebuffer* size in pixels (correct for DPI)
        int fbW, fbH; glfwGetFramebufferSize(win, &fbW, &fbH);

        glm::mat4 proj, view;

        if (!minimap) {
            glViewport(0, 0, fbW, fbH);
            float aspect = (fbH > 0) ? (fbW / float(fbH)) : (16.0f/9.0f);
            proj = glm::perspective(glm::radians(70.0f), aspect, 0.05f, 400.0f);
            view = cam.view();
        } else {
            // World extents (plus padding so the whole maze is visible)
            float w_world = mz.W * cell;
            float h_world = mz.H * cell;
            float pad     = cell * 0.5f; // world-units padding around
            float worldW  = w_world + 2 * pad;
            float worldH  = h_world + 2 * pad;
            float aspectWorld = worldW / worldH;

            // Use up to 90% of the framebuffer size to leave even margins
            int availW = (int)std::floor(fbW * 0.90f + 0.5f);
            int availH = (int)std::floor(fbH * 0.90f + 0.5f);

            int vpW, vpH;
            if (availW / float(availH) > aspectWorld) {
                vpH = availH;
                vpW = (int)std::floor(vpH * aspectWorld + 0.5f);
            } else {
                vpW = availW;
                vpH = (int)std::floor(vpW / aspectWorld + 0.5f);
            }

            // Perfectly center in the framebuffer
            int vpX = (fbW - vpW) / 2;
            int vpY = (fbH - vpH) / 2;
            glViewport(vpX, vpY, vpW, vpH);

            // Ortho projection exactly covering the padded maze
            proj = glm::ortho(-pad, w_world + pad,  h_world + pad, -pad,  0.1f,  200.0f);

            // Camera straight above center, looking down -Y (so +Z is down on screen)
            float cx = w_world * 0.5f, cz = h_world * 0.5f;
            view = glm::lookAt(glm::vec3(cx, 50.0f, cz),
                               glm::vec3(cx,  0.0f, cz),
                               glm::vec3(0.0f, 0.0f, -1.0f));
        }

        // ---- Draw scene / map ----
        glUseProgram(prog);
        if (!minimap) {
            glUniform3f(uLightDir, 0.5f, 1.0f, 0.4f);      // normal 3D lighting
        } else {
            glUniform3f(uLightDir, 0.0f, -1.0f, 0.0f);     // straight-down light for clearer map
        }

        glBindVertexArray(cube.vao);
        for (const auto& inst : instances){
            glm::mat4 MVP = proj * view * inst.model;
            glm::mat3 Nrm = glm::inverseTranspose(glm::mat3(inst.model));
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(inst.model));
            glUniformMatrix3fv(uNormalMat, 1, GL_FALSE, glm::value_ptr(Nrm));
            glUniform3f(uColor, inst.color.r, inst.color.g, inst.color.b);
            glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
        }

        // ---- Player marker on minimap (big, bright, always on top) ----
        if (minimap) {
            glm::mat4 M(1.0f);
            float markerSize = 0.6f * cell;   // 60% of a cell
            float markerH    = 0.3f;          // above the floor

            M = glm::translate(M, glm::vec3(cam.pos.x, markerH, cam.pos.z));
            M = glm::scale(M, glm::vec3(markerSize, markerH, markerSize));

            glm::mat4 MVP = proj * view * M;
            glm::mat3 Nrm = glm::inverseTranspose(glm::mat3(M));

            // Draw on top of everything
            glDisable(GL_DEPTH_TEST);

            glUniformMatrix4fv(uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniformMatrix3fv(uNormalMat, 1, GL_FALSE, glm::value_ptr(Nrm));

            glUniform3f(uColor, 1.0f, 0.2f, 0.9f); // bright magenta
            glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);

            glEnable(GL_DEPTH_TEST);
        }

        glBindVertexArray(0);
        glfwSwapBuffers(win);
        glfwPollEvents();
    } // <-- end while loop

    cube.destroy();
    glDeleteProgram(prog);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}