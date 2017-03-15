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
		index1 = (row-1)*table.stride+col;
		break;
	case DOWN:
		index1 = (row+1)*table.stride+col;
		break;
	case LEFT:
		index1 = row*table.stride+(col-1);
		break;
	case RIGHT:
		index1 = row*table.stride+(col+1);
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
		index1 = (row-1)*table.stride+col;
		break;
	case DOWN:
		index1 = (row+1)*table.stride+col;
		break;
	case LEFT:
		index1 = row*table.stride+(col-1);
		break;
	case RIGHT:
		index1 = row*table.stride+(col+1);
		break;
	}
	deviceSwapElements(&(table.elements[index0]), &(table.elements[index1]));
	return true;
}

/*
	EN: Determines if a rotation with center (col, row) is valid.
	ES: Determina si una rotación con centro (col, row) es válida.
*/
bool rotationIsValid(int row, int col, Table table)
{
	if(row <= 0 || row >= table.height || col <= 0 || col >= table.width)
		return false;
	return true;
}

/*
	EN: Checks if a rotation in valid, and if it is, it goes on and rotates the jewels.
	ES: Comprueba si la rotación es válida, y si lo es, pasa a rotar las joyas.
*/
bool rotate(int row, int col, Table table)
{
	if(rotationIsValid(row, col, table))
	{
		int leftUpMostCorner = table.getElement(row-1, col-1);
		int centerUpMostSpot = table.getElement(row-1, col);
		table.setElement(table.getElement(row+1, col-1), row-1, col-1);
		table.setElement(table.getElement(row+1, col+1), row+1, col-1);
		table.setElement(table.getElement(row-1, col+1), row+1, col+1);
		table.setElement(leftUpMostCorner, row-1, col+1);
		table.setElement(table.getElement(row-1, col), row, col-1);
		table.setElement(table.getElement(row, col-1), row+1, col);
		table.setElement(table.getElement(row+1, col), row, col+1);
		table.setElement(centerUpMostSpot, row+1, col);
		return true;
	}
	return false;
}

/*
	EN: Returns the relative position between two points of the table. 
	curRow: current row, which is being used as a basis for the check.
	curCol: current column, which is being used as a basis for the check.
	row: reference row, which is being used as a basis for rotation.
	col: reference column, which is being used as a basis for rotation.
	ES: Devuelve la posición relativa entre dos puntos del tablero.
	curRow: fila actual, usada como base para la comprobación.
	curCol: columna actual, usada como base para la comprobación.
	row: fila de referencia, que se usará como base para la rotación.
	col: columna de referencia, que se usará como base para la rotación.
*/
__device__ int getRelativePosition(int curRow, int curCol, int row, int col)
{
	int xDistance = curCol - col;
	int yDistance = curRow - row;
	if(xDistance == -1 && yDistance == -1)
		return 0;
	if(xDistance == 0 && yDistance == -1)
		return 1;
	if(xDistance == 1 && yDistance == -1)
		return 2;
	if(xDistance == 1 && yDistance == 0)
		return 3;
	if(xDistance == 1 && yDistance == 1)
		return 4;
	if(xDistance == 0 && yDistance == 1)
		return 5;
	if(xDistance == -1 && yDistance == 1)
		return 6;
	if(xDistance == -1 && yDistance == 0)
		return 7;
	if(xDistance == 0 && yDistance == 0)
		return 8;
	return -1;
}

/*
	EN: Rotates the 3x3 subtable centered in (col, row). First, if check if the rotation is valid,
	then determines which point is being handled by the current thread and makes the appropriate changes.
	ES: Rota un subtablero de 3x3, centrado en (col, row). Primero, comprueba si la rotación es válida,
	entonces determina qué punto está siendo manejado por el hilo actual y hace los cambios apropiados.
*/
__device__ bool rotateDeviceGlobal(int row, int col, Table table)
{
	int curRow = threadIdx.y, curCol = threadIdx.x;
	int relPos = getRelativePosition(curRow, curCol, row, col);
	int newVal;
	switch(relPos)
	{
	case 0:
		newVal = table.getElementDevice(curRow, curCol+2);
		break;
	case 2:
		newVal = table.getElementDevice(curRow+1, curCol+1);
		break;
	case 3:
		newVal = table.getElementDevice(curRow+2, curCol);
		break;
	case 4:
		newVal = table.getElementDevice(curRow+1, curCol-1);
		break;
	case 5:
		newVal = table.getElementDevice(curRow, curCol-2);
		break;
	case 6:
		newVal = table.getElementDevice(curRow-1, curCol-1);
		break;
	case 7:
		newVal = table.getElementDevice(curRow-2, curCol);
		break;
	case 8:
		newVal = table.getElementDevice(curRow-1, curCol+1);
		break;
	}
	__syncthreads();
	table.setElementDevice(newVal, curRow, curCol);
	return true;
}

/*
	EN: Rotates the 3x3 subtable centered in (col, row). First, if check if the rotation is valid,
	then determines which point is being handled by the current thread and makes the appropriate changes.
	ES: Rota un subtablero de 3x3, centrado en (col, row). Primero, comprueba si la rotación es válida,
	entonces determina qué punto está siendo manejado por el hilo actual y hace los cambios apropiados.
*/
__device__ bool rotateDeviceBlock(int row, int col, Table table)
{
	int curRow = blockIdx.y*blockDim.y+threadIdx.y, curCol = blockIdx.x*blockDim.x+threadIdx.x;
	int relPos = getRelativePosition(curRow, curCol, row, col);
	int newVal;
	switch(relPos)
	{
	case 0:
		newVal = table.getElementDevice(curRow, curCol+2);
		break;
	case 2:
		newVal = table.getElementDevice(curRow+1, curCol+1);
		break;
	case 3:
		newVal = table.getElementDevice(curRow+2, curCol);
		break;
	case 4:
		newVal = table.getElementDevice(curRow+1, curCol-1);
		break;
	case 5:
		newVal = table.getElementDevice(curRow, curCol-2);
		break;
	case 6:
		newVal = table.getElementDevice(curRow-1, curCol-1);
		break;
	case 7:
		newVal = table.getElementDevice(curRow-2, curCol);
		break;
	case 8:
		newVal = table.getElementDevice(curRow-1, curCol+1);
		break;
	}
	__syncthreads();
	table.setElementDevice(newVal, curRow, curCol);
	return true;
}

/*
	EN: Rotates the 3x3 subtable centered in (col, row). First, if check if the rotation is valid,
	then determines which point is being handled by the current thread and makes the appropriate changes.
	ES: Rota un subtablero de 3x3, centrado en (col, row). Primero, comprueba si la rotación es válida,
	entonces determina qué punto está siendo manejado por el hilo actual y hace los cambios apropiados.
*/
__device__ bool rotateDeviceShared(int row, int col, Table table)
{
	int curRow = blockIdx.y*blockDim.y+threadIdx.y, curCol = blockIdx.x*blockDim.x+threadIdx.x;
	int relPos = getRelativePosition(curRow, curCol, row, col);
	int newVal;
	switch(relPos)
	{
	case 0:
		newVal = table.getElementDevice(curRow, curCol+2);
		break;
	case 2:
		newVal = table.getElementDevice(curRow+1, curCol+1);
		break;
	case 3:
		newVal = table.getElementDevice(curRow+2, curCol);
		break;
	case 4:
		newVal = table.getElementDevice(curRow+1, curCol-1);
		break;
	case 5:
		newVal = table.getElementDevice(curRow, curCol-2);
		break;
	case 6:
		newVal = table.getElementDevice(curRow-1, curCol-1);
		break;
	case 7:
		newVal = table.getElementDevice(curRow-2, curCol);
		break;
	case 8:
		newVal = table.getElementDevice(curRow-1, curCol+1);
		break;
	}
	__syncthreads();
	table.setElementDevice(newVal, curRow, curCol);
	return true;
}

/*
	EN: Function in charge of erasing a set of elements in a row in CPU. It moves them up, 
	effectively dragging everything above them down, and then randomizes that set of elements.
	ES: Función encargada de eliminar un grupo de elementos dispuestos en forma de fila en CPU.
	En primer lugar los desplaza ascendentemente, hacienedo bajar todo lo que haya sobre ellos,
	y entonces genera valores aleatorios para sustituir los actuales.
*/
void eraseRow(int startRow, int startCol, int length, Table t)
{
	while(moveIsValid(startRow, startCol, moves_t.UP, t))
	{
		for(int i=0; i<length; i++)
		{
			move(startRow, startCol+i, moves_t.UP, t);
		}
	}
	t.recreateRegion(startRow, startCol, length, 1);
}

/*
	EN: Function in charge of erasing a set of elements in a column in CPU. It moves them left,
	effectively dragging those in the positions they now occupy to the right, and then randomizes
	that set of elements.
	ES: Función encargada de eliminar un grupo de elementos dispuestos en forma de columna en CPU.
	En primer lugar desplaza los elementos hacia la izquierda, efectivamente arrastrando los elementos
	en las posiciones que ahora ocupan hacia la derecha, y entonces recrea esa región con valores aleatorios.
*/
void eraseCol(int startRow, int startCol, int length, Table t)
{
	while(moveIsValid(startRow, startCol, moves_t.LEFT, t))
	{
		for(int i=0; i<length; i++)
		{
			move(startRow+i, startCol, moves_t.LEFT, t);
		}
	}
	t.recreateRegion(startRow, startCol, 1, length);
}

__device__ void eraseRowDevice(int startRow, int startCol, int length, Table t)
{
	int row = threadIdx.y, col = threadIdx.x;
	if(row != startRow || col < startCol || col > startCol+length) __syncthreads();
	else
	{
		while(moveIsValid(row, col, moves_t.UP, t))
			move(row, col, moves_t.UP, t);
		t.recreateRegion(row, col, 1, 1);
		__syncthreads();
	}
}

__device__ void eraseColDevice(int startRow, int startCol, int length, Table t)
{
	int row = threadIdx.y, col = threadIdx.x;
	if(col != startCol || row < startRow || row > startRow+length) __syncthreads();
	else
	{
		while(moveIsValid(row, col, moves_t.LEFT, t))
			move(row, col, moves_t.LEFT, t);
		t.recreateRegion(row, col, 1, 1);
		__syncthreads();
	}
}

/*
	EN: Detects what rows and columns should be eliminated.
*/
int[] detectEliminationPattern(Table t, int row, int col)
{
	int count = 0;
	int baseXrow = -1, baseYrow = -1, 
	for(int i=0; i<t.width-1; i++)
	{
		if(t.getElement(row, i) == t.getElement(row, i+1) && t.getElement(row, i) == t.getElement(row, col))
			count++;
		else if(i>=col) break;
		else count = 0;
	}
	count = 0;
	for(int i=0; i<t.height-1; i++)
	{
		if(t.getElement(i, col) == t.getElement(i+1, col) && t.getElement(i, col) == t.getElement(row, col))
			count++;
		else if(i>=row) break;
		else count = 0;
	}
}