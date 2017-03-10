#ifndef GAME_DATA_STRUCTS
#define GAME_DATA_STRUCTS
#include <stdio.h>
#include <stdlib.h>

/*
	EN: Enum defining number-jewel color associations.
	ES: Enum que define las asociaciones entre número y color de joya.
*/
enum jewels{BLUE = 1, RED = 2, ORANGE = 3, GREEN = 4, BROWN = 5, YELLOW = 6, BLACK = 7, WHITE = 8};
/*
	EN: Table struct. It contains information regarding current game status, as well as a variety of functions to help handle it.
	ES: Struct de Table. Contiene información referente al estado actual del juego, además de múltiples funciones para ayudar a manejarlo.
*/

typedef struct table_t{
	int width, height, stride, difficulty;
	int* elements;
	
	/*
		EN: Sets the value of the specified element.
		ES: Establece el valor del elemento especificado.
	*/
	void setElement(int value, int row, int col)
	{
		elements[row*stride+col] = value;
	}
	
	/*
		EN: Retrieves the value of the specified element.
		ES: Recupera el valor del elemento especificado.
	*/
	int getElement(int row, int col)
	{
		return elements[row*stride+col];
	}
	
	/*
		EN: Creates a random table status; used when initializing.
		ES: Crea un estado de tablero aleatorio. Usado cuando se inicializa.
	*/
	void randomize(int gDifficulty)
	{
		for(int i=0; i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				switch(difficulty)
				{	
				case 1:
					setElement(1+rand()%4, i, j);
					break;
				case 2:
					setElement(1+rand()%6, i, j);
					break;
				case 3:
					setElement(1+rand()%8, i, j);
					break;
				}
			}
		}
		difficulty = gDifficulty;
	}
	
	/*
		EN: Creates a new empty table with the established width and height.
		ES: Crea un nuevo tablero con la anchura y altura establecidos.
	*/
	void initialize(int tWidth, int tHeight, int difficulty)
	{
		width = tWidth;
		height = tHeight;
		stride = width;
		elements = (int*)calloc(width*height, sizeof(int));
		//randomize(difficulty);
	}
	
	/*
		EN: Allocates memory for the table in the target device.
		ES: Asigna memoria al tablero en el dispositivo.
	*/
	void deviceInitialize(int tWidth, int tHeight)
	{
		width = tWidth;
		height = tHeight;
		stride = width;
		cudaMalloc(&elements, width*height*sizeof(int));
	}
	
	/*
		EN: Prints the contents of the table to stdout.
		ES: Dibuja el contenido del tablero en el output standard.
	*/
	void print()
	{
		for(int i=0; i<height; i++)
			{
				for(int j=0; j<width; j++)
					printf("%i\t", elements[i*stride+j]);
				printf("\n");
			}
	}
	
	/*
		EN: Retrieves a given subtable of region of the table.
		ES: Recupera un subtablero o region dada del tablero. 
	*/
	struct table_t getSubTable(int startRow, int startCol, int sWidth, int sHeight)
	{
		struct table_t newTable;
		newTable.initialize(sWidth, sHeight);
		free(newTable.elements);
		newTable.elements = &(elements[startRow*stride+startCol]);
		return newTable;
	}
	
	/*
		EN: Recreates a given region of the table.
		ES: Recrea una región dada del tablero.
	*/
	void recreateRegion(int startRow, int startCol, int rWidth, int rHeight)
	{
		Table region = getSubTable(startRow, startCol, rWidth, rHeight);
		region.randomize(difficulty);
	}
	
	/*
		EN: Frees the memory used by the pointers in the table.
		ES: Libera la memoria utilizada por los punteros del tablero.
	*/
	void destroy()
	{
		free(elements);
	}
	
	/*
		EN: Frees the memory used in the device by the table.
		ES: Libera la memoria utilizada por los punteros del tablero en el dispositivo.
	*/
	void deviceDestroy()
	{
		cudaFree(elements);
	}
} Table;

#endif