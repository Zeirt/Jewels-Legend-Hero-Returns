#ifndef SAVE_HELPER_H
#define SAVE_HELPER_H

#include "data_structures.cuh"

/*
	EN: Definition of functions for data persistency. Explanation will be provided in implementation file.
	ES: Definici?n de funciones de persistencia de datos. Se proveeran explicaciones en el archivo de implementaci?n.
*/

bool saveData(int width, int height, bool isManual, Table table);
bool loadData(int* width, int* height, bool* isManual, Table* table);

#endif