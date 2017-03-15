#include "data_structures.cuh"
#include "game_helper.h"
#include "save_helper.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
	EN: Parses arguments (argc, args) and changes the other arguments for the parsed values. 
	Returns false if there was an error. Returns true if game can continue.
	ES: Procesa argumentos (argc, args) y cambia los otros argumentos por los valores procesados. 
	Devuelve false si hubo un error. Devuelve true si el juego puede seguir.
*/
bool argumentParser(int argc, char* args[], int* width, int* height, bool* isManual, int* difficulty) {
	if (argc < 5) {
		//Tell user how to run it and exit
		printf("Uso: jewels.exe m/a dificultad(1-3) numColumnas numFilas \n");
		return false;
	}
	if (strcmp("a", args[1]) == 0) {
		*isManual = false;
	}
	else if (strcmp("m", args[1]) == 0) {
		*isManual = true;
	}
	else {
		printf("Error: Modo de ejecucion debe ser manual <m> o automatico <a> \n");
		return false;
	}
	char *ptr;
	*difficulty = strtol(args[2], &ptr, 10);
	if ((*difficulty < 1) || (*difficulty > 3)) {
		printf("Error: La dificultad debe ser entre 1 y 3\n");
		return false;
	}
	*width = (int)strtol(args[3], &ptr, 10);
	*height = (int)strtol(args[4], &ptr, 10);
	if ((width <= 0) || (height <= 0)) {
		printf("Error: filas y columnas deben ser numeros mayores que 0 \n");
		return false;
	}
	return true;
}

/*
	EN: Parses the input for a turn in the manual mode. Recognices and launches bombs, normal exchange, and exit command.
	Returns true if input is valid and processed. Returns false if input is invalid and nothing was done.
	ES: Procesa el comando del usuario en el modo manual. Reconoce y lanza bombas, intercambios normales y comando de salida.
	Devuelve true si el comando es valido y se ha procesado. Devuelve false si el comando no es valido y no se ha hecho nada.
*/
bool gameInputParser(bool *exit, Table *table) {
	int inputInt, inputInt2;
	char* bufferInput;
	char *pEnd;
	scanf("%s", &bufferInput);
	//Parse input: if asked for a bomb
	if (strcmp("9", bufferInput) == 0) {
		fflush(stdin);
		scanf("%s", bufferInput);
		printf("\nHas pedido una bomba. Inserta tipo (1,2,3)\n");
		//Get bomb type
		if ((strcmp("1", bufferInput) == 0) || (strcmp("2", bufferInput) == 0) || (strcmp("3", bufferInput) == 0)) {
			inputInt = (int)strtol(bufferInput, &pEnd, 10);
			switch (inputInt) {
			case 1: //Bomb asking for col
				fflush(stdin);
				scanf("%s", &bufferInput);
				inputInt2 = (int)strtol(bufferInput, &pEnd, 10);
				//Summon bomb
				eraseCol(0, inputInt2, table->height, *table);
				break;
			case 2: //Bomb asking for row
				fflush(stdin);
				scanf("%s", &bufferInput);
				inputInt2 = (int)strtol(bufferInput, &pEnd, 10);
				eraseRow(inputInt2, 0, table->width, *table);
				break;
			default: //Swap bomb
					 //Summon bomb
				int w = table->width/3, h=table->height/3;
				for(int i=0; i<h; i += 3)
				{
					for(int j=0; j<w; j+=3)
					{
						rotate(i+1, j+1, *table);
					}
				}
				break;
			}
		}
		else {
			printf("Comando incorrecto. Prueba otra vez.\n");
			return false;
		}
	}//Parse input: not a bomb
	else {
		inputInt = (int)strtol(bufferInput, &pEnd, 10);
		if (inputInt != 0) {
			//inputInt is col, inputInt2 is row
			fflush(stdin);
			scanf("%s", &bufferInput);
			inputInt2 = (int)strtol(bufferInput, &pEnd, 10);
			if (inputInt2 != 0) {
				//Call exchange
				printf("\nSeleccione movimiento:\n\t1. ARRIBA\n\t2.ABAJO\n\t3.IZQUIERDA\n\t4.DERECHA\n");
				int movement = -1;
				scanf("%i", &movement);
				switch(movement)
				{
				case 1:
					move(inputInt2, inputInt, UP, *table);
					break;
				case 2:
					move(inputInt2, inputInt, DOWN, *table);
					break;
				case 3:
					move(inputInt2, inputInt, LEFT, *table);
					break;
				case 4:
					move(inputInt2, inputInt, RIGHT, *table);
					break;
				default: printf("Movimiento inválido. Prueba otra vez.\n");
					return false;
				}
			}
			else {
				printf("Comando incorrecto. Prueba otra vez.\n");
				return false;
			}
		}
		else { //Not a number. Are you exiting?
			if (strcmp("q", bufferInput) == 0) {
				*exit = true;
			}
			else {
				printf("Comando incorrecto. Prueba otra vez.\n");
				return false;
			}
		}
	}
	return true;
}