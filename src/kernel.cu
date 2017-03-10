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
	bool success;
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
	char* bufferTabla;
	table.initialize(width, height, difficulty); //We initialize the table and then load the data if possible
	table.deviceInitialize(width, height, difficulty);
	dataSuccess = loadData(&width, &height, &isManual, &difficulty, bufferTabla);
	if (dataSuccess) {
		printf("Se ha cargado la partida anterior.\n");
		//TODO: When load function is created in table, load the string there
	}
	else {
		printf("No se ha podido cargar la partida anterior. Empezando una nueva.\n");
		//TODO: When new initialize is created, launch it
	}
	//Two different kinds of play. Select one or another depending on mode chosen (manual or auto)
	if (isManual) { //Manual play
		while (!exit) {
			table.print();
			//TODO: Add elimination of lines when it's implemented (as much as they can)
			do {
				success = gameInputParser(&exit, &table);
			} while (!success);
			//saveData(width, height, isManual, difficulty, table);
		}
	}
	else { //Auto play
		while (!exit) {
			//Function autoplay
		}
	}
}