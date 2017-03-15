#ifndef GAME_HELPER_H
#define GAME_HELPER_H

#include "data_structures.cuh"

/*
	EN: Definition of possible moves as constants. They will be used in checks.
	ES: Definición de posibles movimientos como constantes. Se usarán para comprobaciones.
*/
typedef enum{UP, DOWN, LEFT, RIGHT} moves_t;

/*
	EN: Declaration of game management functions. Further explanations will be provided
	on the implementation.
	ES: Declaración de funciones de manejo de juego. Se proveerán explicaciones detalladas
	en la implementación.
*/
void swapElements(int*, int*);
__device__ void deviceSwapElements(int*, int*);
bool moveIsValid(int, int, moves_t, Table);
__device__ bool deviceMoveIsValid(int, int, moves_t, Table); 
bool move(int, int, moves_t, Table);
__device__ bool deviceMove(int, int, moves_t, Table);
bool rotationIsValid(int row, int col, Table table);
bool rotate(int row, int col, Table table);
__device__ int getRelativePosition(int curRow, int curCol, int row, int col);
__device__ bool rotateDeviceGlobal(int row, int col, Table table);
__device__ bool rotateDeviceBlock(int row, int col, Table table);
__device__ bool rotateDeviceShared(int row, int col, Table table);
void eraseRow(int startRow, int startCol, int length, Table t);
void eraseCol(int startRow, int startCol, int length, Table t);
__global__ void eraseRowDevice(int startRow, int startCol, int length, Table t);
__global__ void eraseColDevice(int startRow, int startCol, int length, Table t);
void detectEliminationPattern(Table t, int row, int col, int storage[6]);
__global__ void detectEliminationPatternDevice(Table t);
void seekAndDestroy(Table t, int row, int col);
void seekAndDestroyDevice(Table t, int row, int col);
void getBestMove(Table t, int* move, int* row, int* col);

#endif