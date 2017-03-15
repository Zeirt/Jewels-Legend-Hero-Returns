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
	while(moveIsValid(startRow, startCol, UP, t))
	{
		for(int i=0; i<length; i++)
		{
			move(startRow, startCol+i, UP, t);
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
	while(moveIsValid(startRow, startCol, LEFT, t))
	{
		for(int i=0; i<length; i++)
		{
			move(startRow+i, startCol, LEFT, t);
		}
	}
	t.recreateRegion(startRow, startCol, 1, length);
}

/*
	EN: Function in charge of erasing a set of elements in a row in GPU. It moves them up, 
	effectively dragging everything above them down, and then randomizes that set of elements.
	ES: Función encargada de eliminar un grupo de elementos dispuestos en forma de fila en GPU.
	En primer lugar los desplaza ascendentemente, hacienedo bajar todo lo que haya sobre ellos,
	y entonces genera valores aleatorios para sustituir los actuales.
*/
__global__ void eraseRowDevice(int startRow, int startCol, int length, Table t)
{
	int row = threadIdx.y, col = threadIdx.x;
	if(row != startRow || col < startCol || col > startCol+length) __syncthreads();
	else
	{
		while(deviceMoveIsValid(row, col, UP, t))
			deviceMove(row, col, UP, t);
		__syncthreads();
	}
}

/*
	EN: Function in charge of erasing a set of elements in a column in GPU. It moves them left,
	effectively dragging those in the positions they now occupy to the right, and then randomizes
	that set of elements.
	ES: Función encargada de eliminar un grupo de elementos dispuestos en forma de columna en GPU.
	En primer lugar desplaza los elementos hacia la izquierda, efectivamente arrastrando los elementos
	en las posiciones que ahora ocupan hacia la derecha, y entonces recrea esa región con valores aleatorios.
*/
__global__ void eraseColDevice(int startRow, int startCol, int length, Table t)
{
	int row = threadIdx.y, col = threadIdx.x;
	if(col != startCol || row < startRow || row > startRow+length) __syncthreads();
	else
	{
		while(deviceMoveIsValid(row, col, LEFT, t))
			deviceMove(row, col, LEFT, t);
		__syncthreads();
	}
}

/*
	EN: Detects what rows and columns should be eliminated.
	ES: Detecta las filas y columnas a eliminar.
*/
void detectEliminationPattern(Table t, int row, int col, int storage[6])
{
	int count = 0;
	for(int i=0; i<t.width-1; i++)
	{
		if(t.getElement(row, i) == t.getElement(row, i+1) && t.getElement(row, i) == t.getElement(row, col))
			{
				count++;
				if(count == 0){
					storage[0] = row;
					storage[1] = i;
				}
		}
		else if(i>=col) break;
		else count = 0;
	}
	storage[2] = count;
	count = 0;
	for(int i=0; i<t.height-1; i++)
	{
		if(t.getElement(i, col) == t.getElement(i+1, col) && t.getElement(i, col) == t.getElement(row, col))
			{
				count++;
				if(count == 0){
					storage[3] = i;
					storage[4] = col;
				}
		}
		else if(i>=row) break;
		else count = 0;
	}
	storage[5] = count;
}

/*
	EN: Detects if a given point of the matrix should be eliminated and marks it as such.
	ES: Detecta si un punto de la matriz debe ser eliminado y lo marca como tal.
*/
__global__ void detectEliminationPatternDevice(Table t, bool* elimination)
{
	int count = 0;
	int row = threadIdx.y, col = threadIdx.x;
	for(int i=col; i<t.width-1; i++)
	{
		if(t.getElementDevice(row, i) != t.getElementDevice(row, i+1)) break;
		count++;
		if(count>= 3)
		{
			elimination[row*t.width+i] = true;
			elimination[row*t.width+col] = true;
		}
	}
	count = 0;
	for(int i=row; i<t.height-1; i++)
	{
		if(t.getElementDevice(i, col) != t.getElementDevice(i+1, col)) break;
		count++;
		if(count>= 3)
		{
			elimination[i*t.width+col] = true;
			elimination[row*t.width+col] = true;
		}
	}
}

/*
	EN: Determines, using (row, col) as a base, what points should be eliminated.
	ES: Determina, usando (row, col) como base, qué puntos deben ser eliminados.
*/
void seekAndDestroy(Table t, int row, int col)
{
	int eliminations[6];
	eliminations[0] = eliminations[1] = eliminations[3] = eliminations[4] = -1;
	detectEliminationPattern(t, row, col, eliminations);
	if(eliminations[0] > -1 && eliminations[1] > -1 && eliminations[2] > 0)
	{
		eraseRow(eliminations[0], eliminations[1], eliminations[2], t);
	}

	if(eliminations[3] > -1 && eliminations[4] > -1 && eliminations[5] > 0)
	{
		eraseCol(eliminations[3], eliminations[4], eliminations[5], t);
	}
}

/*
	EN: Determines, using (row, col) as a base, what points should be eliminated using GPU.
	ES: Determina, usando (row, col) como base, qué puntos deben ser eliminados usando GPU.
*/
void seekAndDestroyDevice(Table t, int row, int col)
{
	dim3 dimGrid(1,1);
	dim3 dimBlock(t.width, t.height);
	Table deviceCopy;
	deviceCopy.deviceInitialize(t.width, t.height, t.difficulty);
	cudaMemcpy(deviceCopy.elements, t.elements, t.width*t.height*sizeof(int), cudaMemcpyHostToDevice);
	bool* eliminationsDevice;
	cudaMalloc(&eliminationsDevice, t.width*t.height*sizeof(bool));
	detectEliminationPatternDevice<<<dimGrid, dimBlock>>>(deviceCopy, eliminationsDevice);
	bool* eliminations = (bool*) malloc(t.width*t.height*sizeof(bool));
	cudaMemcpy(eliminations, eliminationsDevice, t.width*t.height*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(eliminationsDevice);
	if(eliminations[row*t.width+col])
	{
		int lX=0, lY=0, baseXrow = row, baseYcol = col;
		for(int i=0; i<t.width; i++)
		{
			if(!eliminations[row*t.width+i] && i>col) break;
			else{
				lX = 0;
				continue;
			}
			if(i<baseXrow) baseXrow = i;
			lX++;
		}
		for(int i=0; i<t.height; i++)
		{
			if(!eliminations[i*t.width+col]) break;
			else {
				lY = 0;
				continue;
			}
			if(i < baseYcol) baseYcol = i;
			lY++;
		}
		eraseRowDevice<<<dimGrid, dimBlock>>>(row, baseXrow, lX, deviceCopy);
		eraseColDevice<<<dimGrid, dimBlock>>>(baseYcol, row, lY, deviceCopy);
		cudaMemcpy(t.elements, deviceCopy.elements, t.width*t.height*sizeof(int), cudaMemcpyDeviceToHost);
		t.recreateRegion(row, baseXrow, lX, 1);
		t.recreateRegion(col, baseYcol, 1, lY);
	}
	deviceCopy.destroy();
}

/*
	EN: Determines best from a row level POV.
	ES: Determina el mejor movimiento desde un punto de vista de fila.
*/
void getBestMove(Table t, int* move, int* row, int* col)
{
	int bestXrow=0, bestYrow=0, bestRowLength=0;
	int bestRowMove;

	for(int i=0; i<t.height; i++)
	{
		int curLength = 0;
		int curMove = LEFT;
		for(int j=0; j<t.height-1; j++)
		{
			if(t.getElement(i, j) == t.getElement(i, j+1))
				curLength++;
			else if (i>0)
				{
					if(t.getElement(i, j) == t.getElement(i-1, j+1))
				{
					curLength++;
					curMove = DOWN;
					}
			}
			else if(i<t.height-1)
			{
				if(t.getElement(i,j) == t.getElement(i+1, j+1))
				{
					curLength++;
					curMove = UP;
				}
			}
			if(curMove == LEFT) curLength = 0;
			else if(curLength > bestRowLength)
			{
				bestRowLength = curLength;
				bestRowMove = curMove;
				switch(bestRowMove)
				{
				case UP:
					bestXrow = i-1;
					bestYrow = j+1;
					break;
				case DOWN:
					bestXrow = i+1;
					bestYrow = j+1;
					break;
				}
			}
		}
	}

	*move = bestRowMove;
	*row = bestYrow;
	*col = bestXrow;
}