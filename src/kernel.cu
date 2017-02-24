#include "data_structures.cuh"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char* args[])
{
	/*
	Es: Bloque de aceptación de argumentos. Sale si no tiene suficientes argumentos o son del tipo incorrecto.
	En: Argument acceptance block. Exits if it doesn't have enough arguments or they're the wrong type.
	*/
	if (argc < 4) {
		//Decirle al usuario como arrancar y salir || Tell user how to run it and exit
		printf("Uso: jewels.exe <m>/<a> numColumnas numFilas \n");
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
	int width = strtol(args[2], &ptr, 10);
	int height = strtol(args[3], &ptr, 10);
	if ((width <= 0) || (height <= 0)) {
		printf("Error: filas y columnas deben ser numeros mayores que 0 \n");
		system("PAUSE");
		return 1;
	}
	//PRUEBA DE ARGUMENTOS. IMPRIMIMOS
	printf("Mode: %s\nWidth:%d\nHeight:%d\n", args[1], width, height);
	system("PAUSE");
	return 0;
}