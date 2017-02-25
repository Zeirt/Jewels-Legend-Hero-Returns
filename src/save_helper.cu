#include <stdio.h>
#include "game_helper.h"

/*
	EN: Saves parameters of the current execution and the table in a .dat file. Returns true if success, false if fail.
	ES: Guarda parametros de la ejecución actual y la tabla en un archivo .dat. Devuelve true si hay éxito, false si falla.
*/
bool saveData(int width, int height, bool isManual, char* table){
	FILE *saveData;

	saveData = fopen("save.dat", "W");
	//Open it as empty file for writing. That means we overwrite old save data.
	if (saveData == NULL) return false; //File couldn't open.

	fwrite(&width, sizeof(int), 1, saveData);
	fwrite(&height, sizeof(int), 1, saveData);
	fwrite(&isManual, sizeof(bool), 1, saveData);
	fwrite(&table, sizeof(table), 1, saveData); //THIS MAY NEED TWEAKING

	//Finish writing and close the file
	fclose(saveData);
	return true;
}

/*
	EN: Loads the data saved by saveData() in order to put it back in the program to re-start from there. 
	ES: Carga los datos guardados por saveData() para insertarlos de nuevo en el programa y reanudar desde ese estado.
*/
bool loadData(int* width, int* height, bool* isManual, char* table){
	FILE *saveData;

	saveData = fopen("save.dat", "r");
	//Open existing file for reading. If it doesn't exist, exit the function
	if (saveData == NULL) return false;

	fread(&width, sizeof(int), 1, saveData);
	fread(&height, sizeof(int), 1, saveData);
	fread(&isManual, sizeof(bool), 1, saveData);
	//I know we settled for passing this to the table directly, but truth to be told, it's unsafe. Returning string
	fread(&table, ((*width * *height) * sizeof(char)), 1, saveData);

	//Finish reading and close the file
	fclose(saveData);
	return true;
}