#include "data_structures.cuh"
#include "save_helper.h"
#include "game_helper.h"
#include "game_loop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char* args[])
{
	int height, width, difficulty;
	bool isManual, success;
	//Parse arguments
	success = argumentParser(argc, args, &width, &height, &isManual, &difficulty);
	if (!success) { //If returned fail, exit
		system("PAUSE");
		return 1;
	}
	//PRUEBA DE ARGUMENTOS. IMPRIMIMOS
	printf("Mode: %s\nWidth:%d\nHeight:%d\n", args[1], width, height);
	//PRUEBA DE ESCRITURA Y LECTURA
	char* placeholder = "hella";
	success = saveData(width, height, isManual, difficulty, placeholder);
	if (success) printf("Escritura de fichero correcta\n");
	else {
		printf("Ha fallado la escritura del fichero\n");
		system("PAUSE");
		return 1;
	}
	success = loadData(&width, &height, &isManual, &difficulty, placeholder);
	if (success) {
		printf("Escritura de fichero correcta\n");
		printf("Mode: %s\nWidth:%d\nHeight:%d\n", args[1], width, height);
		printf("Table string; %s", &placeholder);
		system("PAUSE");
		return 0;
	}
	else {
		printf("Ha fallado la lectura del fichero\n");
		system("PAUSE");
		return 1;
	}
	//Begin game loop
	bool exit, dataSuccess;
	exit = false;
	Table table;
	char* bufferTabla = (char*)malloc((width*height+4)*sizeof(char));
	table.initialize(width, height, difficulty); //We initialize the table and then load the data if possible
	dataSuccess = loadData(&width, &height, &isManual, &difficulty, bufferTabla);
	if (dataSuccess) {
		printf("Se ha cargado la partida anterior.\n");
		isManual = table.loadSaveString(bufferTabla);
	}
	else {
		printf("No se ha podido cargar la partida anterior. Empezando una nueva.\n");
		table.randomize(difficulty);
	}
	//Two different kinds of play. Select one or another depending on mode chosen (manual or auto)
	if (isManual) { //Manual play
		while (!exit) {
			table.print();
			do {
				success = gameInputParser(&exit, &table);
			} while (!success);
			table.createSaveString(bufferTabla, isManual);
			saveData(width, height, isManual, difficulty, bufferTabla);
		}
	}
	else { //Auto play
		while (!exit) {
			//Function autoplay
			table.print();
			int movement;
			int row, col;
			getBestMove(table, &movement, &row, &col);
			switch(movement)
			{
			case 0:
				move(row, col, UP, table);
				break;
			case 1:
				move(row, col, DOWN, table);
				break;
			}
			seekAndDestroy(table, row, col);
			table.createSaveString(bufferTabla, isManual);
			saveData(width, height, isManual, difficulty, bufferTabla);
		}
	}
	table.destroy();
	free(bufferTabla);
}