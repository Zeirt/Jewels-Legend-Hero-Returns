#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include <fstream>

//Función que genera una jewel al azar
int createJewel(int difficulty) {
	srand(time(NULL));
	switch (difficulty) {
	case 1: {
		int randomJewel = rand() % 4 + 1;
		return randomJewel;
	}
	case 2: {
		int randomJewel = rand() % 6 + 1;
		return randomJewel;
	}
	case 3: {
		int randomJewel = rand() % 8 + 1;
		return randomJewel;
	}
	}
	return -1;
}

void initialTablePopulation(float *table, int difficulty, int width, int height) {
	srand(time(NULL));
	for (int i = 0; i < height*width; i++) {
		switch (difficulty) {
		case 1: {
			int randomJewel = rand() % 4 + 1;
			table[i] = randomJewel;
			break;
		}
		case 2: {
			int randomJewel = rand() % 6 + 1;
			table[i] = randomJewel;
			break;
		}
		case 3: {
			int randomJewel = rand() % 8 + 1;
			table[i] = randomJewel;
			break;
		}
		}
	}
}

void printTable(float* table, int width, int height) {
	for (int i = height - 1; i >= 0; i--) {
		std::cout<<std::endl;;
		for (int j = 0; j < width; j++) {
			printf("%d ", (int)table[j + i*width]);
		}
	}
	std::cout<<std::endl;;
}

void eraseJewels(float* table, float* jewelsToErase, int difficulty, int width, int height) {
	int max = 0;

	if (height >= width) max = height;
	else max = width;

	int end = 0;
	bool altered = false;

	//Calcula cuál es el valor escrito de entre aquellos a eliminar, para poder encontrar potenciales huecos
	for (int i = 0; i < max * 2; i++) {
		if (jewelsToErase[i] < 0) {
			end = i;
			altered = true;
			break;
		}
	}

	//Todos los valeros están escritos
	if (!altered) end = max * 2;

	srand(time(NULL));

	if (jewelsToErase[0] != jewelsToErase[2]) {
		for (int y = jewelsToErase[1]; y < height; y++) {
			for (int x = jewelsToErase[0]; x <= jewelsToErase[end - 2]; x++) {
				if (y + 1 < height) {
					table[x + (y)*(width)] = table[x + (y + 1)*width];
					switch (difficulty) {
					case 1: {
						int randomJewel = rand() % 4 + 1;
						table[x + (y+1)*width] = randomJewel;
						break;
					}
					case 2: {
						int randomJewel = rand() % 6 + 1;
						table[x + (y+1)*width] = randomJewel;
						break;
					}
					case 3: {
						int randomJewel = rand() % 8 + 1;
						table[x + (y+1)*width] = randomJewel;
						break;
					}
					}
				}
				else {
					switch (difficulty) {
					case 1: {
						int randomJewel = rand() % 4 + 1;
						table[x + y*width] = randomJewel;
						break;
					}
					case 2: {
						int randomJewel = rand() % 6 + 1;
						table[x + y*width] = randomJewel;
						break;
					}
					case 3: {
						int randomJewel = rand() % 8 + 1;
						table[x + y*width] = randomJewel;
						break;
					}
					}
				}
			}
		}
	}else{
		int spot = jewelsToErase[0] + jewelsToErase[1] * width;
		float value = table[spot];
		for (int y = jewelsToErase[1]; y < height; y++) {
			for (int x = jewelsToErase[0]; x <= jewelsToErase[end - 2]; x++) {
				if (y < height) {
					if (y >= jewelsToErase[end-2]) {
						table[x + (y-end/2)*(width)] = table[x + (y)*width];
						switch (difficulty) {
						case 1: {
							int randomJewel = rand() % 4 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						case 2: {
							int randomJewel = rand() % 6 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						case 3: {
							int randomJewel = rand() % 8 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						}
					}
					else {
						switch (difficulty) {
						case 1: {
							int randomJewel = rand() % 4 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						case 2: {
							int randomJewel = rand() % 6 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						case 3: {
							int randomJewel = rand() % 8 + 1;
							table[x + (y)*width] = randomJewel;
							break;
						}
						}
					}
				}
			}
		}
	}
}

void manualModeTableAnalysis(int difficulty, float* table, int width, int height, int x, int y) {
	int max = 0;
	int size = width*height;

	if (height >= width) max = height;
	else max = width;

	//Eliminaremos, como mucho, max jewels, almacenando su posición (x, y) en el proceso
	float* jewelsToErase = (float*)malloc(2 * max * sizeof(float));

	for (int i = 0; i < max; i++) {
		jewelsToErase[i] = -1;
	}

	int leftHndPotentialJewels = 0;
	int rightHndPotentialJewels = 0;
	//Exploración por la izquierda
	if ((x - 1 + y*width >= 0) && table[x - 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x - i + y*width >= 0) && (x -i>=0) && table[x - i + y*width] == table[x + y*width]) {
			leftHndPotentialJewels++;
			i++;
		}
	}

	//Exploración por la derecha
	if ((x + 1 + y*width <= size) && table[x + 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x + i + y*width <= size) && (x + i < width) && table[x + i + y*width] == table[x + y*width]) {
			rightHndPotentialJewels++;
			i++;
		}
	}

	//Existe la posibilidad de eliminar horizontalmente
	if (1 + leftHndPotentialJewels + rightHndPotentialJewels >= 3) {
		int stride = 0;

		for (int j = leftHndPotentialJewels; j >= (1); j--) {
			jewelsToErase[stride] = x - j;
			jewelsToErase[stride + 1] = y;
			stride += 2;
		}

		jewelsToErase[leftHndPotentialJewels*2] = x;
		jewelsToErase[leftHndPotentialJewels*2+1] = y;

		stride = 2;
		for (int k = 1; k <= rightHndPotentialJewels; k++) {
			jewelsToErase[stride + leftHndPotentialJewels*2] = x + k;
			jewelsToErase[stride + leftHndPotentialJewels*2 + 1] = y;
			stride += 2;
		}
	}
	else {	//Exploración de la columna
		int potentialJewelsOver = 0;
		int potentialJewelsBelow = 0;

		//Exploración por debajo
		if ((x + (y - 1)*width >= 0) && table[x + (y - 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y - i)*width >= 0) && table[x + (y - i)*width] == table[x + y*width]) {
				potentialJewelsBelow++;
				i++;
			}
		}

		//Exploración por encima
		if ((x + 1 + y*width <= size) && table[x + (y + 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y + i)*width <= size) && table[x + (y + i)*width] == table[x + y*width]) {
				potentialJewelsOver++;
				i++;
			}
		}

		//Existe la posibilidad de realizar una eliminación vertical
		if (1 + potentialJewelsBelow + potentialJewelsOver >= 3) {

			int stride = 0;
			for (int j = potentialJewelsBelow; j >= (1); j--) {
				jewelsToErase[stride] = x;
				jewelsToErase[stride + 1] = y - j;
				stride += 2;
			}

			jewelsToErase[potentialJewelsBelow*2] = x;
			jewelsToErase[potentialJewelsBelow*2+1] = y;

			stride = 2;
			for (int k = 1; k <= potentialJewelsOver; k++) {
				jewelsToErase[stride + potentialJewelsBelow*2] = x;
				jewelsToErase[stride + potentialJewelsBelow*2 + 1] = y + k;
				stride += 2;
			}
		}
	}

	eraseJewels(table, jewelsToErase, difficulty, width, height);
	free(jewelsToErase);
}

void switchSpots(float* table, int jewel1X, int jewel1Y, int direction, int width, int height, int selection, int difficulty) {
	int jewel2_x = jewel1X;
	int jewel2_y = jewel1Y;
	switch (direction)
	{
	case 1: //Arriba
	{
		jewel2_y += 1;
		break;
	}
	case 2: //Abajo
	{
		jewel2_y -= 1;
		break;
	}
	case 3: //Izquierda
	{
		jewel2_x -= 1;
		break;
	}
	case 4: //Derecha
	{
		jewel2_x += 1;
		break;
	}
	}
	int aux1;

	aux1 = table[jewel2_x + jewel2_y*width];

	table[jewel2_x + jewel2_y*width] = table[jewel1X + jewel1Y*width];
	table[jewel1X + jewel1Y*width] = aux1;

	if (selection == 2)
		manualModeTableAnalysis(difficulty, table, width, height, jewel2_x, jewel2_y);
}

//Función de análisis del tablero en modo automático en CPU
void autoModeTableAnalisis(int difficulty, float* table, int width, int height) {
	int max = 0;
	int size = width*height;
	int rightHndPotentialJewels = 0;

	if (height >= width) max = height;
	else max = width;

	//Se eliminana, como mucho, max jewels, almacenando su posición (x, y)
	float* jewelsToErase = (float*)malloc(2 * max * sizeof(float));

	//Tablero auxiliar para seleccionar el mejor caso
	float* auxTable = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < max; i++) {
		jewelsToErase[i] = -1;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			rightHndPotentialJewels = 0;

			//Si tiene por la derecha
			if ((x + 2) < width) {
				if (((x + 2) + y*width <= size) && table[x + 2 + y*width] == table[x + y*width]) {
					int i = 2;
					while ((x + i + y*width <= size) && table[x + i + y*width] == table[x + y*width]) {
						rightHndPotentialJewels++;
						i++;
					}

					auxTable[x + y*width] = rightHndPotentialJewels + 1;
				}
				else {
					auxTable[x + y*width] = 1;
				}
			}
			else {
				auxTable[x + y*width] = 1;
			}
		}
	}

	int bestX = 0;
	int bestY = 0;
	int bestValue = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (auxTable[x + y*width] > bestValue) {
				bestX = x;
				bestY = y;
				bestValue = auxTable[x + y*width];
			}
		}
	}

	switchSpots(table, bestX, bestY, 4, width, height, 1, difficulty);

	//Se puede realizar eliminaciones
	if (bestValue >= 3) {
		jewelsToErase[0] = bestX;
		jewelsToErase[1] = bestY;

		int stride = 2;

		for (int j = 1; j <= (bestValue); j++) {
			jewelsToErase[stride] = bestX + j;
			jewelsToErase[stride + 1] = bestY;
			stride += 2;
		}
	}

	eraseJewels(table, jewelsToErase, difficulty, width, height);
	free(jewelsToErase);
	free(auxTable);
}

bool preloadGame(int& width, int& height, int& difficulty, char* row)
{
	std::ifstream fwidth("width.txt");
	if (!fwidth.is_open())
	{
		std::cerr << "ERROR: Archivo de guardado (width.txt) no encontrado." << std::endl;
		return false;
	}
	fwidth >> width;
	fwidth.close();

	std::ifstream fheight("height.txt");

	if (!fheight.is_open())
	{
		std::cout << "ERROR: Archivo de guardado (height.txt) no encontrado." << std::endl;
		return false;
	}
	fheight >> height;
	fheight.close();
	std::ifstream fdifficulty("difficulty.txt");

	if (!fdifficulty.is_open())
	{
		std::cout << "ERROR: Archivo de guardado (difficulty.txt) no encontrado." << std::endl;
		return false;
	}
	fdifficulty >> difficulty;
	fdifficulty.close();
	std::ifstream fLoad(row);
	if (!fLoad.is_open())
	{
		std::cout << "ERROR: Archivo de guardado no encontrado." << std::endl;
		return false;
	}
	fLoad.close();
	return true;
}

void loadFile(int width, int height, float*  table, char* row)
{
	int aux;
	char* array = (char*)malloc(width*height + 1);
	std::ifstream fLoad(row);
	fLoad.getline(array, width*height + 1);

	for (int i = 0; i < width*height; i++)
	{
		aux = (array[i] - 48);
		table[i] = (float)aux;
	}
	free(array);
	fLoad.close();

}

void saveFile(float* table, int width, int height, int difficulty, char* row)
{
	//Sistema de saveFile

	std::ofstream rowWidth;
	rowWidth.open("width.txt");
	rowWidth.clear();
	rowWidth << width;
	rowWidth.close();
	std::ofstream rowHeight;
	rowHeight.open("height.txt");
	rowHeight.clear();
	rowHeight << height;
	rowHeight.close();
	std::ofstream rowDifficulty;
	rowDifficulty.open("difficulty.txt");
	rowDifficulty.clear();
	rowDifficulty << difficulty;
	rowDifficulty.close();

	std::ofstream saveF;
	saveF.open(row);
	saveF.clear();

	for (int index = 0; index < width*height; index++)
	{
		saveF << table[index];
	}
	saveF.close();

}

void rowBomb(float* table, int width, int height, int difficulty, int row) {

	for (int i = 0; (i + row) < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if ((i + row + 1) < height)
			{
				
				table[(i + row)*width + j] = table[(i + row + 1)*height + j];
			}
			else {
				table[(i + row)*width + j] = createJewel(difficulty);
			}
		}
	}
}

void columnBomb(float* table, int width, int height, int difficulty, int column) {

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; (column - j) > 0; j++)
		{
			if ((column - j - 1) < 0)
			{
				table[(i*width) + (column - j)] = createJewel(difficulty);
			}
			else {
				table[(i*width) + (column - j)] = table[(i*height) + (column - j - 1)];
			}
		}
	}
}

void pivotBomb(float* table, int width, int height, int row, int column)
{
	float aux[9];
	int index = 0;
	for (int j = column - 1; j <= column + 1; j++)
	{
		for (int i = row + 1; i >= row - 1; i--)
		{
			aux[index] = table[i*width + j];
			index++;
		}
	}
	index = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int icolumn = 0; icolumn < 3; icolumn++)
		{
			table[(i + row - 1)*width + (column - 1) + icolumn] = aux[index];
			index++;
		}
	}
}

int main(int argc, char** argv) {
	//Matriz dinámica de elementos de tipo float, con tamaño width*height
	int width;
	int height;
	int difficulty;
	char mode;
	bool autoPlay = true;
	int size;
	char saveF[9] = "save.txt";
	bool found = false;
	int selection;

	float *table;
	//Entrada de la configuración del juego
	if (argc == 1)
	{
		std::cout << "Anchura del tablero: ";
		std::cin >> width;

		std::cout << "Altura del tablero: ";
		std::cin >> height;

		std::cout << "Elija dificultad:"<<std::endl<<"\t1.-\tFacil"<<std::endl<<"\t2.-\tMedia"<<std::endl<<"\t3.-\tDificil\n";
		std::cin >> difficulty;

		std::cout << "Jugar automaticamente?"<<std::endl<<"\t1.-\tSI"<<std::endl<<"\t2.-\tNO"<<std::endl;
		std::cin >> selection;
	}
	else
	{
		mode = argv[1][1];
		difficulty = atoi(argv[2]);
		width = atoi(argv[3]);
		height = atoi(argv[4]);

		switch (mode) {
		case 'a': {selection = 1; 
			break; }
		case 'm': {selection = 2; 
			break; }
		default: std::cerr<<"ERROR: Valor de modo inválido. Por favor, inserte 'a' o 'm' como modo de juego en la línea de comandos."<<std::endl; 
			return -1;
		}
	}
	
	bool playing = true;

	/* Establecer autoPlay como mode de juego */
	
	size = width*height;
	table = (float*)malloc(size * sizeof(float));

	//Se inicializa la matriz
	initialTablePopulation(table, difficulty, width, height);

	//Bucle principal del juego
	while (playing) {

		printTable(table, width, height);

		int jewel1X = 0;
		int jewel1Y = 0;
		int command = 0;

		std::cout << "Acción a realizar:"<<std::endl;
		std::cout << "1.-\tIntercambiar Jewels"<<std::endl;
		std::cout << "2.-\tGuardar partida"<<std::endl;
		std::cout << "3.-\tCargar partida"<<std::endl;
		std::cout << "9.-\tUsar una bomba"<<std::endl;
		std::cout << "0.-\tSalir"<<std::endl;
		std::cout << "Inserte seleccion: ";

		std::cin >> command;

		switch (command) {
		case 0: {
			free(table);
			return 0;
			break;
		}
		case 1: {

			if (selection == 2)
			{
				std::cout << "Posicion de la jewel (el primer valor es 0):"<<std::endl;
				std::cout << "\tColumna: ";
				std::cin >> jewel1X;
				std::cout << "\tFila: ";
				std::cin >> jewel1Y;

				if (!((jewel1X < width) && (jewel1X >= 0) && (jewel1Y < height) && (jewel1Y >= 0))) {
					printf("Posicion invalida.\n");
					continue;
				}

				int direction = 0;
				std::cout << "Direccion del movimiento:<<"<<std::endl<<"\t1.-\tArriba"<<std::endl<<"\t2.-\tAbajo"<<std::endl<<"\t3.-\tIzquierda"<<std::endl<<"\t4.-\tDerecha"<<std::endl;
				std::cin >> direction;

				if (direction > 4 && direction > 1) {
					printf("Movimiento invalido.\n");
					continue;
				}
				else {
					switch (direction)
					{
					case 1: //Arriba
					{
						if (jewel1Y == height)
						{
							printf("No se puede realizar el intercambio especificado.\n");
							continue;
						}
						break;
					}
					case 2: //Abajo
					{
						if (jewel1Y == 0)
						{
							printf("No se puede realizar el intercambio especificado.\n");
							continue;
						}
						break;
					}
					case 3: //Izquierda
					{
						if (jewel1X == 0)
						{
							printf("No se puede realizar el intercambio especificado.\n");
							continue;
						}
						break;
					}
					case 4: //Derecha
					{
						if (jewel1X == width - 1)
						{
							printf("No se puede realizar el intercambio especificado.\n");
							continue;
						}
						break;
					}
					}
				}
				// Desplaza las jewels como se ha indicado
				switchSpots(table, jewel1X, jewel1Y, direction, width, height, selection, difficulty);

			}
			else if (selection == 1)
			{
				// Modo automático
				autoModeTableAnalisis(difficulty, table, width, height);
			}
			break;
		}
		case 2: {
			saveFile(table, width, height, difficulty, saveF);
			std::cout << "Guardado realizado."<<std::endl;
			break;
		}
		case 3: {

			//Precargado del tablero
			int found = preloadGame(width, height, difficulty, saveF);
			size = width*height;
			if (found)
			{
				free(table);
				table = (float*)malloc(size * sizeof(float));

				// Cargado del tablero
				loadFile(width, height, table, saveF);
				std::cout << "Juego automatico?"<<std::endl<<"\t1.-\tSI"<<std::endl<<"\t2.-\tNO"<<std::endl;
				std::cin >> selection;
				std::cout << "Se ha cargado el estado del tablero:"<<std::endl;
			}
			else {
				std::cout << "No hay partidas guardadas."<<std::endl;
			}
			break;



		}
		case 9: {
			int bomb = 0;
			int row = 0; int column = 0;
			std::cout << "Seleccione el tipo de bomba:"<<std::endl;

			// Bombas según la dificultad 
			switch (difficulty) {
			case 1: {
				std::cout << "1.-\tBomba sobre fila"<<std::endl;
				std::cout << "\tSeleccion: "<<std::endl;
				std::cin >> bomb;

				if (bomb != 1)
				{
					printf("Tipo de bomba inexistente.\n");
					continue;
				}
				std::cout << "X: ";
				std::cin >> row;
				rowBomb(table, width, height, difficulty, row);
				break;
			}
			case 2: {
				std::cout << "1.-\tBomba sobre fila"<<std::endl;
				std::cout << "2.-\tBomba sobre columna"<<std::endl;
				std::cout << "\tSeleccion: "<<std::endl;
				std::cin >> bomb;

				if (bomb < 1 && bomb > 2)
				{
					printf("Tipo de bomba inexistente.\n");
					continue;
				}
				switch (bomb) {
				case 1:
				{
					std::cout << "X: ";
					std::cin >> row;
					rowBomb(table, width, height, difficulty, row);
					break;
				}
				case 2:
				{
					std::cout << "Y: ";
					std::cin >> column;
					columnBomb(table, width, height, difficulty, column);
					break;
				}
				}
				break;
			}
			case 3: {
				std::cout << "1.-\tBomba sobre fila"<<std::endl;
				std::cout << "2.-\tBomba sobre columna"<<std::endl;
				std::cout << "3.-\tBomba de rotacion 3x3"<<std::endl;
				std::cout << "\tSeleccion: "<<std::endl;
				std::cin >> bomb;

				if (bomb < 1 && bomb > 3)
				{
					printf("Tipo de bomba inexistente.\n");
					continue;
				}
				switch (bomb) {
				case 1:
				{
					std::cout << "X: ";
					std::cin >> row;
					rowBomb(table, width, height, difficulty, row);
					break;
				}
				case 2:
				{
					std::cout << "Y: ";
					std::cin >> column;
					columnBomb(table, width, height, difficulty, column);
					break;
				}
				case 3:
				{
					for (int row = 1; row < width; row += 3)
					{
						for (int column = 1; column < height; column += 3)
						{
							if (!((row - 1) < 0 || (row + 1) >= height || (column - 1) < 0 || (column + 1) >= width))
							{
								pivotBomb(table, width, height, row, column);
							}
						}
					}
					break;
				}
				}

				break;
			}
			}
			break;
		}
		}
	}
	free(table);
	return 0;
}