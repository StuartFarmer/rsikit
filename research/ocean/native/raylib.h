/* Headless raylib surface used by pinned g2048.h; rendering and input are off. */
#pragma once
#include <stdbool.h>
typedef struct Color { unsigned char r, g, b, a; } Color;
enum { KEY_LEFT_SHIFT, KEY_UP, KEY_W, KEY_DOWN, KEY_S, KEY_LEFT, KEY_A,
       KEY_RIGHT, KEY_D, KEY_ESCAPE };
static inline bool IsWindowReady(void) { return false; }
static inline bool IsKeyDown(int key) { return false; }
static inline bool IsKeyPressed(int key) { return false; }
static inline void InitWindow(int w, int h, const char *title) {}
static inline void SetTargetFPS(int fps) {}
static inline void CloseWindow(void) {}
static inline void BeginDrawing(void) {}
static inline void EndDrawing(void) {}
static inline void ClearBackground(Color color) {}
static inline void DrawRectangle(int x, int y, int w, int h, Color color) {}
static inline void DrawText(const char *text, int x, int y, int size, Color color) {}
