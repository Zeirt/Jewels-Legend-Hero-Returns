#include "game_helper.h"

/*
	EN: Simple swapper function. Takes in two pointers and exchanges their contents.
	ES: Función swapper simple. Recibe dos punteros e intercambia sus contenidos.
*/
void swapElements(int* elementA, int* elementB)
{
	int tmp = *elementA;
	*elementA = *elementB;
	*elementB = tmp;
}

/*
	EN: Swapper function designed to run on GPU. Takes in two pointers and exchanges their contents.
	ES: Función swapper diseñada para correr en GPU. Recibe dos punteros e intercambia sus contenidos.
*/
__device__ void deviceSwapElements(int* elementA, int* elementB)
{
	int tmp = *elementA;
	*elementA = *elementB;
	*elementB = *elementA;
}

/*
	EN: Checks if a move is valid in CPU. Using row, column and move information, it decides if the new position 
	lies within the boundaries of the game table.
	ES: Comprueba si un movimiento es válido en CPU. Usando información de fila, columna y movimiento, decide
	si la nueva posición se encuentra dentro de los límites del tablero.
*/
bool moveIsValid(int row, int col, moves_t move, Table t)
{
	switch(move)
	{
	case UP:
		return row-1 >= 0;
		break;
	case DOWN:
		return row+1 < t.height;
		break;
	case LEFT:
		return col-1 >= 0;
		break;
	case RIGHT:
		return col+1 < t.width;
		break;
	default:
		return false;
	}
}

/*
	EN: Checks if a move is valid in GPU. Using row, column and move information, it decides if the new position 
	lies within the boundaries of the game table.
	ES: Comprueba si un movimiento es válido en GPU. Usando información de fila, columna y movimiento, decide
	si la nueva posición se encuentra dentro de los límites del tablero.
*/
__device__ bool deviceMoveIsValid(int row, int col, moves_t move, Table t)
{
	switch(move)
	{
	case UP:
		return row-1 >= 0;
		break;
	case DOWN:
		return row+1 < t.height;
		break;
	case LEFT:
		return col-1 >= 0;
		break;
	case RIGHT:
		return col+1 < t.width;
		break;
	default:
		return false;
	}
}

/*
	EN: Moves pieces around in CPU. First, it checks if the move is valid; then, it moves the pieces.
	This function returns true if the move was successful, or false otherwise.
	ES: Mueve las piezas en CPU. Primero, comprueba si el movimiento es válido; entonces, mueve las piezas.
	Esta función devuelve true si el movimiento es exitoso, o falso en otro caso.
*/
bool move(int row, int col, moves_t move, Table table)
{
	if(!moveIsValid(row, col, move, table)) return false;
	int index0 = row*table.stride+col, index1;
	switch(move)
	{
	case UP:
		index1 = (row-1)*table+col;
		break;
	case DOWN:
		index1 = (row+1)*table+col;
		break;
	case LEFT:
		index1 = row*table+(col-1);
		break;
	case RIGHT:
		index1 = row*table+(col+1);
		break;
	}
	swapElements(&(table.elements[index0]), &(table.elements[index1]));
	return true;
}

/*
	EN: Moves pieces around in GPU. First, it checks if the move is valid; then, it moves the pieces.
	This function returns true if the move was successful, or false otherwise.
	ES: Mueve las piezas en GPU. Primero, comprueba si el movimiento es válido; entonces, mueve las piezas.
	Esta función devuelve true si el movimiento es exitoso, o falso en otro caso.
*/
__device__ bool deviceMove(int row, int col, moves_t move, Table table)
{
	if(!deviceMoveIsValid(row, col, move, table)) return false;
	int index0 = row*table.stride+col, index1;
	switch(move)
	{
	case UP:
		index1 = (row-1)*table+col;
		break;
	case DOWN:
		index1 = (row+1)*table+col;
		break;
	case LEFT:
		index1 = row*table+(col-1);
		break;
	case RIGHT:
		index1 = row*table+(col+1);
		break;
	}
	deviceSwapElements(&(table.elements[index0]), &(table.elements[index1]));
	return true;
}