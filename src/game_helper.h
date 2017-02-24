#ifndef GAME_HELPER_H
#define GAME_HELPER_H
#include <stdbool.h>
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

#endif