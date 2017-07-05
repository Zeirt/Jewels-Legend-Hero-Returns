#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include <fstream>

//funcion para generar una jewel aleatoria, como la generacion inicial.
/* Funciones para generar gemas aleatorias */
/* Iniciador de seeds */
__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

/* Crear jewel usando globalState */
__device__ float generate(curandState* globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}

/* Funcion para generarJewel en CUDA */
__device__ int createJewelCUDA(curandState* globalState, int ind, int difficulty)
{
	switch (difficulty) {
	case 1:
	{
		return (int)1 + generate(globalState, ind) * 4;
	}
	case 2: {
		return (int)1 + generate(globalState, ind) * 6;
	}
	case 3: {
		return (int)1 + generate(globalState, ind) * 8;
	}
	}
	return -1;
}

/* Funcion para inicializar la matriz de gemas */
__global__ void initialTablePopulation(float *table, int difficulty, int width, int height, curandState* globalState) {
	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;
	if (trow < height)
	{
		if (tcolumn < width)
		{
			table[trow*width + tcolumn] = createJewelCUDA(globalState, trow * width + tcolumn, difficulty);
		}
	}
}
void printTable(float* table, int width, int height) {
	for (int i = height - 1; i >= 0; i--) {
		printf("\n");
		for (int j = 0; j < width; j++) {
			printf("%d ", (int)table[j + i*width]);
		}
	}
	printf("\n");
}

/*Recibe las coordenadas de las jewels a eliminar y mueve las rows que tiene que bajar a partir de ellas, emplea
una copia del table para evitar race conditions*/
__global__ void jewelErasingKernel(float* table_d, float* table_aux_d, float* erasedJewels_d, int difficulty, int width, int height, int end, curandState* globalState) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (erasedJewels_d[0] != erasedJewels_d[2] && tx >= erasedJewels_d[0] && tx <= erasedJewels_d[end - 2] && ty >= erasedJewels_d[1]) {
		if (ty + 1 < height) {

			float value = table_aux_d[tx + (ty + 1)*width];

			table_d[tx + (ty)*(width)] = value;
		}
		else {
			table_d[tx + ty*width] = createJewelCUDA(globalState, tx + ty*width, difficulty);
		}
	}
	else {

		if (ty < height && tx == erasedJewels_d[0] && ty > erasedJewels_d[1]) {

			float value = table_aux_d[tx + (ty)*width];

			table_d[tx + (ty - end / 2)*(width)] = value;

		}

		if (ty >= height - end / 2 && ty < height && tx == erasedJewels_d[0]) {

			table_d[tx + (ty)*width] = createJewelCUDA(globalState, tx + ty*width, difficulty);

		}
	}
}

/*Funcion que prepara y llama el kernel con su mismo nombre, genera todos los datos necesarios*/
void eraseJewels(float* table, float* erasedJewels, int difficulty, int width, int height, curandState* globalState) {
	float *table_d;
	float *erasedJewels_d;
	float *table_aux_d;
	int size = width * height * sizeof(float);
	int max = 0;

	//Para saber que medida es la más grande, ya que no se pueden eliminar más jewels seguidas que esa medida
	if (height >= width) max = height;
	else max = width;

	//table a GPU y la copia del table
	cudaMalloc((void**)&table_d, size);
	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&table_aux_d, size);
	cudaMemcpy(table_aux_d, table, size, cudaMemcpyHostToDevice);

	//Jewels a eliminar a GPU. 2*max ya que cada posicion son dos coordenadas, x e y
	cudaMalloc((void**)&erasedJewels_d, 2 * max * sizeof(float));

	cudaMemcpy(erasedJewels_d, erasedJewels, 2 * max * sizeof(float), cudaMemcpyHostToDevice);

	int end = 0;
	bool altered = false;

	//Calcula cual es el ultimo valor escrito de las jewels a eliminar, ya que puede haber posiciones no escritas
	for (int i = 0; i < max * 2; i++) {
		if (erasedJewels[i] < 0) {
			end = i;
			altered = true;
			break;
		}
	}

	//En caso de que este completamente escrito
	if (!altered) end = max * 2;

	//Configuracion de ejecucion
	dim3 dimBlock(width, height);
	dim3 dimGrid(1, 1);

	jewelErasingKernel <<<dimGrid, dimBlock >>> (table_d, table_aux_d, erasedJewels_d, difficulty, width, height, end, globalState);

	//Se recupera el table actualizado

	cudaMemcpy(table, table_d, size, cudaMemcpyDeviceToHost);

	//Libera memoria
	cudaFree(table_d);
	cudaFree(erasedJewels_d);
	cudaFree(table_aux_d);
}

/*Escribe en un table auxiliar la cantidad de jewels que se eliminarian moviendo una jewel (x,y) hacia la derecha
paralelizable ya que todos los hilos (cada hilo 1 jewel) tienen que expandirse hacia la derecha para ver hasta donde llegarian a eliminar*/
__global__ void autoTableAnalysisKernel(float *table_d, float *aux_d, int difficulty, int width, int height) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int potentialJewelsRight = 0;

	//Si tiene por la derecha
	if ((tx + 2) < width) {
		if (((tx + 2) + ty*width <= height*width) && table_d[tx + 2 + ty*width] == table_d[tx + ty*width]) {
			int i = 2;
			//Se expande
			while ((tx + i + ty*width <= height*width) && table_d[tx + i + ty*width] == table_d[tx + ty*width]) {
				potentialJewelsRight++;
				i++;
			}

			aux_d[tx + ty*width] = potentialJewelsRight + 1;
		}
		else {
			aux_d[tx + ty*width] = 1;
		}
	}
	else {
		aux_d[tx + ty*width] = 1;
	}
}

//Analiza el movimiento manual, usando las coordenadas de la nueva posicion de la jewel selectionada
void manualTableAnalysis(int difficulty, float* table, int width, int height, int x, int y, curandState* globalState) {
	int max = 0;
	int size = width*height;

	if (height >= width) max = height;
	else max = width;

	//Solo se eliminan MAX jewels como mucho, se guardan sus x e y
	float* erasedJewels = (float*)malloc(2 * max * sizeof(float));

	//Se inicializa a -1 àra saber hasta que punto se escribe
	for (int i = 0; i < max; i++) {
		erasedJewels[i] = -1;
	}

	int potentialJewelsRight = 0;

	//Si tiene por la izquierda
	if ((x - 1 + y*width >= 0) && table[x - 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x - i + y*width >= 0) && (x - i >= 0) && table[x - i + y*width] == table[x + y*width]) {
			potentialJewelsRight++;
			i++;
		}
	}

	//Si tiene por la derecha
	if ((x + 1 + y*width <= size) && table[x + 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x + i + y*width <= size) && (x + i < width) && table[x + i + y*width] == table[x + y*width]) {
			potentialJewelsRight++;
			i++;
		}
	}

	//Se pueden eliminar horizontalmente, las coloca en orden para facilitar su eliminacion
	if (1 + potentialJewelsRight + potentialJewelsRight >= 3) {
		int stride = 0;

		for (int j = potentialJewelsRight; j >= (1); j--) {
			erasedJewels[stride] = x - j;
			erasedJewels[stride + 1] = y;
			stride += 2;
		}

		erasedJewels[potentialJewelsRight * 2] = x;
		erasedJewels[potentialJewelsRight * 2 + 1] = y;

		stride = 2;
		for (int k = 1; k <= potentialJewelsRight; k++) {
			erasedJewels[stride + potentialJewelsRight * 2] = x + k;
			erasedJewels[stride + potentialJewelsRight * 2 + 1] = y;
			stride += 2;
		}
	}
	else {	//Analizamos la vertical
		int potentialJewelsAbove = 0;
		int potentialJewelsBelow = 0;

		//Si tiene por abajo
		if ((x + (y - 1)*width >= 0) && table[x + (y - 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y - i)*width >= 0) && table[x + (y - i)*width] == table[x + y*width]) {
				potentialJewelsBelow++;
				i++;
			}
		}

		//Si tiene por arriba
		if ((x + 1 + y*width <= size) && table[x + (y + 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y + i)*width <= size) && table[x + (y + i)*width] == table[x + y*width]) {
				potentialJewelsAbove++;
				i++;
			}
		}

		//Se pueden eliminar
		if (1 + potentialJewelsBelow + potentialJewelsAbove >= 3) {

			int stride = 0;
			for (int j = potentialJewelsBelow; j >= (1); j--) {
				erasedJewels[stride] = x;
				erasedJewels[stride + 1] = y - j;
				stride += 2;
			}

			erasedJewels[potentialJewelsBelow * 2] = x;
			erasedJewels[potentialJewelsBelow * 2 + 1] = y;

			stride = 2;
			for (int k = 1; k <= potentialJewelsAbove; k++) {
				erasedJewels[stride + potentialJewelsBelow * 2] = x;
				erasedJewels[stride + potentialJewelsBelow * 2 + 1] = y + k;
				stride += 2;
			}
		}
	}

	//Las elimina
	eraseJewels(table, erasedJewels, difficulty, width, height, globalState);
	free(erasedJewels);
}

//Intercambia la jewel selectionadas con la jewel en la dirección indicada
void swapSpots(float* table, int jewel1X, int jewel1Y, int direction, int width, int height, int selection, int difficulty, curandState* globalState) {
	int jewel2X = jewel1X;
	int jewel2Y = jewel1Y;
	switch (direction)
	{
	case 1: //Arriba
	{
		jewel2Y += 1;
		break;
	}
	case 2: //Abajo
	{
		jewel2Y -= 1;
		break;
	}
	case 3: //Izquierda
	{
		jewel2X -= 1;
		break;
	}
	case 4: //Derecha
	{
		jewel2X += 1;
		break;
	}
	}
	int aux1;

	aux1 = table[jewel2X + jewel2Y*width];

	table[jewel2X + jewel2Y*width] = table[jewel1X + jewel1Y*width];
	table[jewel1X + jewel1Y*width] = aux1;

	//Analiza el movimiento para ver si se pueden eliminar jewels
	manualTableAnalysis(difficulty, table, width, height, jewel2X, jewel2Y, globalState);
}

//Analiza la mejor opcion y la ejecuta en funcion de lo que devuelve el kernel
void automaticTableAnalysis(int difficulty, float* table, int width, int height, curandState* globalState) {
	float *table_d;
	float *aux_d;
	float *aux;
	//Tamaño del table para asignar memoria
	int size = width * height * sizeof(float);
	int tam = width * height;
	int max = 0;

	if (height >= width) max = height;
	else max = width;

	//Solo se eliminan max jewels, 2 coordenadas por jewel = 2 * max posiciones

	float* erasedJewels = (float*)malloc(2 * max * sizeof(float));
	aux = (float*)malloc(size);

	for (int i = 0; i < max; i++) {
		erasedJewels[i] = -1;
	}

	//Solo se cuenta la jewel que se escoge, sigue siendo menor que 3
	for (int p = 0; p < tam; p++) {
		aux[p] = 1;
	}

	//table a GPU
	cudaMalloc((void**)&table_d, size);

	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	//Auxiliar de conteo a GPU

	cudaMalloc((void**)&aux_d, size);

	cudaMemcpy(aux_d, aux, size, cudaMemcpyHostToDevice);

	//Configuracion de ejecucion
	dim3 dimBlock(width, height);
	dim3 dimGrid(1, 1);

	//Inicio del kernel

	autoTableAnalysisKernel <<<dimGrid, dimBlock >>> (table_d, aux_d, difficulty, width, height);

	//Transfiere el resultado de la GPU al host
	cudaMemcpy(aux, aux_d, size, cudaMemcpyDeviceToHost);

	int x_mejor = 0;
	int y_mejor = 0;
	int valor_mejor = 0;

	//Se busca el movimiento con el mayor numero de jewels eliminadas
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (aux[x + y*width] > valor_mejor) {
				valor_mejor = aux[x + y*width];
				x_mejor = x;
				y_mejor = y;
			}
		}
	}

	//Si se pueden eliminar se ejecuta el movimiento, con lo que ello conlleva
	if (valor_mejor >= 3) {
		swapSpots(table, x_mejor, y_mejor, 4, width, height, 1, difficulty, globalState);
	}
	free(aux);
	free(erasedJewels);
	cudaFree(table_d);
	cudaFree(aux_d);
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

/* Funcion que elimina una row */
__global__ void rowBomb(float* table, int width, int height, int difficulty, int row, curandState* globalState) {

	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;

	if ((trow + row) < height)
	{
		if (tcolumn < width)
		{
			if ((trow + row + 1) == height)
			{
				table[(trow + row)*width + tcolumn] = createJewelCUDA(globalState, (trow * 3 + tcolumn), difficulty);
			}
			else { 
				table[(trow + row)*width + tcolumn] = table[(trow + row + 1)*width + tcolumn];
			}
		}
	}
}

/* Funcion que elimina una column */
__global__ void columnBomb(float* table, int width, int height, int difficulty, int column, curandState* globalState) {

	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;

	if (trow < height)
	{
		if ((tcolumn + column) < width)
		{
			if ((column - tcolumn - 1) < 0)
			{
				table[(trow*width) + (column - tcolumn)] = createJewelCUDA(globalState, (trow * 3 + tcolumn), difficulty);
			}
			else {
				table[(trow*width) + (column - tcolumn)] = table[(trow*width) + (column - tcolumn - 1)];
			}
		}
	}
}

__global__ void pivotBombGPU(float* table, int width, int height, int row, int column)
{
	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;

	if (trow < 3)
	{
		if (tcolumn < 3)
		{
			table[(row + 1 - tcolumn)*width + (column - 1 + trow)] = table[((row + 1) - trow)*width + ((column + 1) - tcolumn)];
		}
	}
}

__global__ void pivotBomb(float* table_d, int width, int height)
{
	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;
	if (trow < height && tcolumn < width) {
		if (!((trow - 1) < 0 || (trow + 1) >= height || (tcolumn - 1) < 0 || (tcolumn + 1) >= width))
		{
			if (trow % 3 == 1 && tcolumn % 3 == 1)
			{
				dim3 dimBlock(3, 3);
				dim3 dimGrid(1, 1);
				//printf(" %i-%i ", trow, tcolumn);
				pivotBombGPU <<<dimGrid, dimBlock >>> (table_d, width, height, trow, tcolumn);
				//__syncthreads();
			}
		}
	}
}
int main(int argc, char** argv) {
	//Matriz de tamaño variable de floats, un array de height*width
	int width;
	int height;
	int difficulty;
	char mode;
	int size;
	char saveF[9] = "save.txt";
	int selection;

	float* table;
	float* table_d;

	curandState* devStates;

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
		default: std::cerr<<"ERROR: Valor de mode inválido. Por favor, inserte 'a' o 'm' como mode de juego en la línea de comandos."<<std::endl; 
			return -1;
		}
	}
	
	bool playing = true;

	size = width*height;

	/* Inicializacion random en CUDA */
	cudaMalloc(&devStates, size * sizeof(curandState));

	/* Creacion de las Seeds */
	setup_kernel <<< 1, size >>> (devStates, unsigned(time(NULL)));

	/* Reservar memoria para table y table_d */
	table = (float*)malloc(size * sizeof(float));
	cudaMalloc((void**)&table_d, size * sizeof(float));

	/* Se inicializa la matriz */
	dim3 dimBlock(width, height);
	dim3 dimGrid(1, 1);
	initialTablePopulation <<<dimGrid, dimBlock >>>(table_d, difficulty, width, height, devStates);
	cudaMemcpy(table, table_d, size * sizeof(float), cudaMemcpyDeviceToHost);

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
			cudaFree(table_d);
			cudaFree(devStates);
			return 0;
		}
				/* Intercambio de jewel */
		case 1: {
			if (selection == 2)
			{
				std::cout << "Posicion de la primera jewel a intercambiar (empiezan en 0)\n";
				std::cout << "column: ";
				std::cin >> jewel1X;
				std::cout << "row: ";
				std::cin >> jewel1Y;

				if (!((jewel1X < width) && (jewel1X >= 0) && (jewel1Y < height) && (jewel1Y >= 0))) {
					printf("Posicion erronea.\n");
					continue;
				}

				int direction = 0;
				std::cout << "Posicion de la jewel (el primer valor es 0):"<<std::endl;
				std::cout << "\tColumna: ";
				std::cin >> jewel1X;
				std::cout << "\tFila: ";
				std::cin >> jewel1Y;

				if (!((jewel1X < width) && (jewel1X >= 0) && (jewel1Y < height) && (jewel1Y >= 0))) {
					printf("Posicion invalida.\n");
					continue;
				}

				std::cout << "direction del movimiento:<<"<<std::endl<<"\t1.-\tArriba"<<std::endl<<"\t2.-\tAbajo"<<std::endl<<"\t3.-\tIzquierda"<<std::endl<<"\t4.-\tDerecha"<<std::endl;
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
				/* Intercambiar posiciones */
				swapSpots(table, jewel1X, jewel1Y, direction, width, height, selection, difficulty, devStates);

			}
			else if (selection == 1)
			{
				/* Analisis automatico */
				automaticTableAnalysis(difficulty, table, width, height, devStates);
			}
			break;
		}
		//Guardar estado de partida
		case 2: {

			saveFile(table, width, height, difficulty, saveF);
			std::cout << "Guardado completado"<<std::endl;
			break;
		}
		//Cargar estado de partida
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
				/* Bombas */
		case 9: {

			int bomb = 0;
			int row = 0; int column = 0;
			std::cout << "Elija una bomba:";

			/* Bombas por tipo de difficulty */
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
				dim3 dimBlock(width, height);
				dim3 dimGrid(1, 1);
				cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
				rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, devStates);
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
					dim3 dimBlock(width, height);
					dim3 dimGrid(1, 1);
					cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
					rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, devStates);
					break;
				}
				case 2:
				{
					std::cout << "Y: ";
					std::cin >> column;
					dim3 dimBlock(width, height);
					dim3 dimGrid(1, 1);
					cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
					columnBomb <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, devStates);
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
					dim3 dimBlock(width, height);
					dim3 dimGrid(1, 1);
					cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
					rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, devStates);
					break;
				}
				case 2:
				{
					std::cout << "Y: ";
					std::cin >> column;
					dim3 dimBlock(width, height);
					dim3 dimGrid(1, 1);
					cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
					columnBomb <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, devStates);
					break;
				}
				case 3:
				{
					dim3 dimBlock(width, height);
					dim3 dimGrid(1, 1);
					cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);
					pivotBomb <<<dimGrid, dimBlock >>>(table_d, width, height);
					break;
				}
				}
				break;
			}
			}
			cudaMemcpy(table, table_d, size * sizeof(float), cudaMemcpyDeviceToHost);
			break;
		}

		}

	}
	free(table);
	cudaFree(table_d);
	cudaFree(devStates);
	return 0;
}