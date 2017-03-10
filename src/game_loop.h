#ifndef GAME_LOOP_H
#define GAME_LOOP_H

#include "data_structures.cuh"
#include "game_helper.h"
#include "save_helper.h"

/*
	EN: Functions aiding the main game. Parse arguments at start and parse the input in manual mode.
	This file has been created to avoid bloating kernel.cu
	ES: Funciones de ayuda al juego principal. Reconoce argumentos al comienzo y comandos en modo manual.
	Este archivo ha sido creado para evitar demasiado codigo en kernel.cu
*/

bool argumentParser(int argc, char* args[], int* width, int* height, bool* isManual, int* difficulty);
bool gameInputParser(bool *exit, Table *table);

#endif