#include "data_structures.cuh"
#include "save_helper.h"
#include "game_helper.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char* args[])
{
	/*
	Es: Bloque de aceptación de argumentos. Sale si no tiene suficientes argumentos o son del tipo incorrecto.
	En: Argument acceptance block. Exits if it doesn't have enough arguments or they're the wrong type.
	*/
	if (argc < 5) {
		//Decirle al usuario como arrancar y salir || Tell user how to run it and exit
		printf("Uso: jewels.exe m/a dificultad(1-3) numColumnas numFilas \n");
		system("PAUSE");
		return 1;
	}
	bool isManual;
	if (strcmp("a", args[1]) == 0) {
		isManual = false;
	}
	else if (strcmp("m", args[1]) == 0) {
		isManual = true;
	}
	else {
		printf("Error: Modo de ejecucion debe ser manual <m> o automatico <a> \n");
		system("PAUSE");
		return 1;
	}
	char *ptr;
	int difficulty = strtol(args[2], &ptr, 10);
	if((difficulty < 1) || (difficulty > 3)){
		printf("Error: La dificultad debe ser entre 1 y 3\n");
		system("PAUSE");
		return 1;
	}
	int width = strtol(args[3], &ptr, 10);
	int height = strtol(args[4], &ptr, 10);
	if ((width <= 0) || (height <= 0)) {
		printf("Error: filas y columnas deben ser numeros mayores que 0 \n");
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

	/*
	Es: Bloque de bucle de juego. Cargará los datos si son necesarios y, basado en eso, inicializa el juego
	En: Game loop block. Loads saved data if necessary and, based on that, initializes the game 
	*/
	bool exit, dataSuccess;
	exit = false;
	Table table;
	char* bufferTabla;
	table.initialize(width, height); //We initialize the table and then load the data if possible
	table.deviceInitialize(width, height);
	dataSuccess = loadData(&width, &height, &isManual, &difficulty, bufferTabla);
	if (dataSuccess) {
		printf("Se ha cargado la partida anterior.\n");
		//TODO: When load function is created in table, load the string there
	}
	else {
		printf("No se ha podido cargar la partida anterior. Empezando una nueva.\n");
		//TODO: When new initialize is created, launch it
	}
	int inputInt;
	char* bufferInput;
	//Two different kinds of play. Select one or another depending on mode chosen (manual or auto)
	if (isManual) {
		while (!exit) {
			table.print();
			//TODO: Add elimination of lines when it's implemented (as much as they can)
			scanf("%d", bufferInput);
			//TODO: Parse input to decide what the user asked for
		}
	}
	else {
		while (!exit) {

		}
	}
}