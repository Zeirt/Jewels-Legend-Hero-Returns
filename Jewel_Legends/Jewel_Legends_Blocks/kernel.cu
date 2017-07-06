#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include <fstream>

//Compruebas las propiedades de la tarjeta gráfica para generar una tesela adecuada, al tiempo que evalua el tamaño del tablero.
int getTileWidth(int width, int height) {
	float minSize = 0;

	if (width > height) minSize = width;
	else minSize = height;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	int maxThreads = properties.maxThreadsPerBlock;

	if (width == height) {	//En caso de una matriz cuadrada, queremos tener más de un bloque
		if (minSize / 32 > 1 && maxThreads == 1024) { //El bloque solo podrá ser de 32x32 si tiene 1024 hilos o más
			return 32;
		}
		else if (minSize / 16 > 1) {
			return 16;
		}
		else if (minSize / 8 > 1) {
			return 8;
		}
		else if (minSize / 4 > 1) {
			return 4;
		}
		else if (minSize / 2 > 1) {
			return 2;
		}
	}
	else {
		if (minSize / 32 >= 1 && maxThreads == 1024) {
			return 32;
		}
		else if (minSize / 16 >= 1) {
			return 16;
		}
		else if (minSize / 8 >= 1) {
			return 8;
		}
		else if (minSize / 4 >= 1) {
			return 4;
		}
		else if (minSize / 2 >= 1) {
			return 2;
		}
	}
	return -1;
}

// Función que inicializa la semilla del generador de números aleatorios de cuda
__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

//Función que crea una jewel utilizando globalState
__device__ float generate(curandState* globalState, int ind)
{
	curandState localState = globalState[ind];
	float rndm = curand_uniform(&localState);
	globalState[ind] = localState;
	return rndm;
}

// Función que crea una jewel aleatoria en CUDA
__device__ int createJewel(curandState* globalState, int ind, int difficulty)
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

// Función que rellena inicialmente el tablero
__global__ void rndmTableInit(float *table, int difficulty, int width, int height, int TILE_WIDTH, curandState* globalState) {
	int trow = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int tcolumn = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (trow < height)
	{
		if (tcolumn < width)
		{
			table[trow*width + tcolumn] = createJewel(globalState, trow * width + tcolumn, difficulty);
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

//Función que, en base a las coordenadas de su hilo de ejecución, determina si la jewel correspondiente ha de ser eliminada y hace descender las filas
//correspondientes en base a ello. Utiliza una copia del tablero para evitar condiciones de carrera.
__global__ void eraseJewelsKernel(float* table_d, float* auxTable_d, float* erasedJewels_d, int difficulty, int width, int height, int end, int TILE_WIDTH, curandState* globalState) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	//Posicion real dentro del table
	tx += block_x * TILE_WIDTH;
	ty += block_y * TILE_WIDTH;

	if (erasedJewels_d[0] != erasedJewels_d[2] && tx >= erasedJewels_d[0] && tx <= erasedJewels_d[end - 2] && ty >= erasedJewels_d[1]) {
		if (ty + 1 < height) {
			float value = auxTable_d[tx + (ty + 1)*width];
			table_d[tx + (ty)*(width)] = value;
		}
		else {
			table_d[tx + ty*width] = createJewel(globalState, tx + ty*width, difficulty);
		}
	}
	else {

		if (ty < height && tx == erasedJewels_d[0] && ty > erasedJewels_d[1]) {
			float value = auxTable_d[tx + (ty)*width];
			table_d[tx + (ty - end / 2)*(width)] = value;
		}

		if (ty >= height - end / 2 && ty < height && tx == erasedJewels_d[0]) {
			table_d[tx + (ty)*width] = createJewel(globalState, tx + ty*width, difficulty);
		}
	}
}

//Función que prepara y ejecuta la eliminación de jewels en GPU.
void eraseJewels(float* table, float* erasedJewels, int difficulty, int width, int height, int TILE_WIDTH, curandState* globalState) {
	float *table_d;
	float *erasedJewels_d;
	float *auxTable_d;
	int size = width * height * sizeof(float);
	int max = 0;

	//Determina cuántas jewels se pueden, como mucho, eliminar en una dirección.
	if (height >= width) max = height;
	else max = width;

	//Envía a GPU el tablero y una copia
	cudaMalloc((void**)&table_d, size);
	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&auxTable_d, size);
	cudaMemcpy(auxTable_d, table, size, cudaMemcpyHostToDevice);

	//Envía las jewels a eliminar a GPU. Se necesita reservar dos veces la cantidad de memoria ocupada
	//por max*sizeof(float) porque cada jewel se identifica mediante dos cifras.
	cudaMalloc((void**)&erasedJewels_d, 2 * max * sizeof(float));

	cudaMemcpy(erasedJewels_d, erasedJewels, 2 * max * sizeof(float), cudaMemcpyHostToDevice);

	int end = 0;
	bool altered = false;

	//Calcula cuál es el último valor escrito de entre todas las jewels a eliminar
	for (int i = 0; i < max * 2; i++) {
		if (erasedJewels[i] < 0) {
			end = i;
			altered = true;
			break;
		}
	}

	if (!altered) end = max * 2;

	//Calcula las dimensiones en función de TILE_WIDTH
	int wdth = ceil(((double)width) / TILE_WIDTH);
	int hght = ceil(((double)height) / TILE_WIDTH);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);
	eraseJewelsKernel <<<dimGrid, dimBlock >>> (table_d, auxTable_d, erasedJewels_d, difficulty, width, height, end, TILE_WIDTH, globalState);

	//Recuperación del nuevo tablero
	cudaMemcpy(table, table_d, size, cudaMemcpyDeviceToHost);

	//Libera memoria
	cudaFree(table_d);
	cudaFree(erasedJewels_d);
	cudaFree(auxTable_d);
}

// Almacena en una tablero auxiliar todas las jewels eliminadas al realizar un movimiento dado desde una posición.
// El paralelismo se consigue realizando una exploración hacia la derecha.
__global__ void automaticTableAnalysisKernel(float *table_d, float *aux_d, int difficulty, int width, int height, int TILE_WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	//Cálculo de la posición real dentro del tablero
	tx += block_x * TILE_WIDTH;
	ty += block_y * TILE_WIDTH;

	int potentialJewelsRight = 0;

	//Exploración por la derecha
	if ((tx + 2) < width) {
		if (((tx + 2) + ty*width <= height*width) && table_d[tx + 2 + ty*width] == table_d[tx + ty*width]) {
			int i = 2;
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

//Análisis del tablero tras un movimiento manual, en base a las nuevas coordenadas de la jewel desplazada
void manualTableAnalysis(int difficulty, float* table, int width, int height, int x, int y, int TILE_WIDTH, curandState* globalState) {
	int max = 0;
	int size = width*height;

	if (height >= width) max = height;
	else max = width;

	//Se realiza la eliminación de jewels, registrando su posición (x, y)
	float* erasedJewels = (float*)malloc(2 * max * sizeof(float));

	//Valores inicalizads a -1 por la posibilidad de que no se pueda eliminar nada
	for (int i = 0; i < max; i++) {
		erasedJewels[i] = -1;
	}

	int potentialJewelsLeft = 0;
	int potentialJewelsRight = 0;

	//Explorar por la izquierda
	if ((x - 1 + y*width >= 0) && table[x - 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x - i + y*width >= 0) && (x - i >= 0) && table[x - i + y*width] == table[x + y*width]) {
			potentialJewelsLeft++;
			i++;
		}
	}

	//Explorar por la derecha
	if ((x + 1 + y*width <= size) && table[x + 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x + i + y*width <= size) && (x + i < width) && table[x + i + y*width] == table[x + y*width]) {
			potentialJewelsRight++;
			i++;
		}
	}

	//Posibilidad de realizar eliminación horizontal
	if (1 + potentialJewelsLeft + potentialJewelsRight >= 3) {
		int stride = 0;

		for (int j = potentialJewelsLeft; j >= (1); j--) {
			erasedJewels[stride] = x - j;
			erasedJewels[stride + 1] = y;
			stride += 2;
		}

		erasedJewels[potentialJewelsLeft * 2] = x;
		erasedJewels[potentialJewelsLeft * 2 + 1] = y;

		stride = 2;
		for (int k = 1; k <= potentialJewelsRight; k++) {
			erasedJewels[stride + potentialJewelsLeft * 2] = x + k;
			erasedJewels[stride + potentialJewelsLeft * 2 + 1] = y;
			stride += 2;
		}
	}
	else {	//Análisis de la vertical
		int potentialJewelsAbove = 0;
		int potentialJewelsBelow = 0;

		//Explorar por debajo
		if ((x + (y - 1)*width >= 0) && table[x + (y - 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y - i)*width >= 0) && table[x + (y - i)*width] == table[x + y*width]) {
				potentialJewelsBelow++;
				i++;
			}
		}

		//Explorar por arriba
		if ((x + 1 + y*width <= size) && table[x + (y + 1)*width] == table[x + y*width]) {
			int i = 1;
			while ((x + (y + i)*width <= size) && table[x + (y + i)*width] == table[x + y*width]) {
				potentialJewelsAbove++;
				i++;
			}
		}

		//Eliminación en la vertical
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

	//Se ejecuta la eliminación
	eraseJewels(table, erasedJewels, difficulty, width, height, TILE_WIDTH, globalState);
	free(erasedJewels);
}

//Desplaza la jewel seleccionada en la dirección indicada. Asume que el movimiento es posible
void swapSpots(float* table, int jewel1X, int jewel1Y, int direction, int width, int height, int selection, int difficulty, int TILE_WIDTH, curandState* globalState) {
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

	//Comprueba si hay eliminaciones posibles
	manualTableAnalysis(difficulty, table, width, height, jewel2X, jewel2Y, TILE_WIDTH, globalState);
}

//Determina y ejecuta el mejor movimiento posible
void automaticTableAnalysis(int difficulty, float* table, int width, int height, int TILE_WIDTH, curandState* globalState) {
	float *table_d;
	float *aux_d;
	float *aux;
	//Tamaño del tablero en memoria
	int size = width * height * sizeof(float);
	int tam = width * height;
	int max = 0;

	if (height >= width) max = height;
	else max = width;

	//Se evaluan las dos coordenadas de cada posición a eliminar

	float* erasedJewels = (float*)malloc(2 * max * sizeof(float));
	aux = (float*)malloc(size);

	for (int i = 0; i < max; i++) {
		erasedJewels[i] = -1;
	}

	for (int p = 0; p < tam; p++) {
		aux[p] = 1;
	}

	//Envía el tablero a GPU
	cudaMalloc((void**)&table_d, size);
	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	
	//Envía una variable auxiliar para conteo a GPU
	cudaMalloc((void**)&aux_d, size);
	cudaMemcpy(aux_d, aux, size, cudaMemcpyHostToDevice);

	//Cálculo de las dimensiones en base a TILE_WIDTH
	int wdth = ceil(((double)width) / TILE_WIDTH);
	int hght = ceil(((double)height) / TILE_WIDTH);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);

	//Ejecución de kernel
	automaticTableAnalysisKernel <<<dimGrid, dimBlock >>> (table_d, aux_d, difficulty, width, height, TILE_WIDTH);

	//Recupera el resultado de GPU a host
	cudaMemcpy(aux, aux_d, size, cudaMemcpyDeviceToHost);

	int bestX = 0;
	int bestY = 0;
	int valor_mejor = 0;

	//Se exploran todos los movimientos en busca del mejor
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (aux[x + y*width] > valor_mejor) {
				valor_mejor = aux[x + y*width];
				bestX = x;
				bestY = y;
			}
		}
	}

	//Se ejecuta el mejor movimiento
	if (valor_mejor >= 3) {
		swapSpots(table, bestX, bestY, 4, width, height, 1, difficulty, TILE_WIDTH, globalState);
	}
	free(aux);
	free(erasedJewels);
	cudaFree(table_d);
	cudaFree(aux_d);
}

bool preload(int& width, int& height, int& difficulty, char* file)
{
	std::ifstream fwidth("width.txt");
	std::ifstream fheight("height.txt");
	std::ifstream fdifficulty("difficulty.txt");
	std::ifstream loadF(file);

	if (!fwidth.is_open())
	{
		std::cout << "ERROR: no existe un archivo save." << std::endl;
		return false;
	}
	if (!fheight.is_open())
	{
		std::cout << "ERROR: no existe un archivo save." << std::endl;
		return false;
	}
	if (!fdifficulty.is_open())
	{
		std::cout << "ERROR: no existe un archivo save." << std::endl;
		return false;
	}
	if (!loadF.is_open())
	{
		std::cout << "ERROR: no existe un archivo save." << std::endl;
		return false;
	}
	fwidth >> width;
	fheight >> height;
	fdifficulty >> difficulty;

	fwidth.close();
	fheight.close();
	fdifficulty.close();
	loadF.close();
	return true;
}

void load(int width, int height, float*  table, char* file)
{
	int aux;
	char* array = (char*)malloc(width*height + 1);
	std::ifstream loadF(file);
	loadF.getline(array, width*height + 1);

	for (int i = 0; i < width*height; i++)
	{
		aux = (array[i] - 48);
		table[i] = (float)aux;
	}
	free(array);
	loadF.close();

}

void save(float* table, int width, int height, int difficulty, char* file)
{
	//Sistema de guardado
	std::ofstream savedFile;
	std::ofstream fileWidth;
	std::ofstream fileHeight;
	std::ofstream fileDifficulty;
	savedFile.open(file);
	fileWidth.open("width.txt");
	fileHeight.open("height.txt");
	fileDifficulty.open("difficulty.txt");

	// Vacía los archivos de guardado
	savedFile.clear();
	fileWidth.clear();
	fileHeight.clear();
	fileDifficulty.clear();

	// Escribe anchura, altura y dificultad
	fileWidth << width;
	fileHeight << height;
	fileDifficulty << difficulty;
	// Escribe el tablero al correspondiente archivo
	for (int index = 0; index < width*height; index++)
	{
		savedFile << table[index];
	}
	savedFile.close();
	fileWidth.close();
	fileHeight.close();
	fileDifficulty.close();
}

// Función que elimina una fila completa
__global__ void rowBomb(float* table, int width, int height, int difficulty, int row, int TILE_WIDTH, curandState* globalState) {
	int trow = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int tcolumn = blockIdx.x*TILE_WIDTH + threadIdx.x;
	float aux;

	if ((trow + row) < height)
	{
		if (tcolumn < width)
		{
			if ((trow + row + 1) == height)
			{
				table[(trow + row)*width + tcolumn] = createJewel(globalState, (trow * 3 + tcolumn), difficulty);
			}
			else {
				aux = table[(trow + row + 1)*width + tcolumn];
				table[(trow + row)*width + tcolumn] = aux;
			}
		}
	}
}

// Función que elimina una columna completa
__global__ void columnBomb(float* table, int width, int height, int difficulty, int column, int TILE_WIDTH, curandState* globalState) {
	int trow = blockIdx.y*TILE_WIDTH +threadIdx.y;
	int tcolumn = blockIdx.x*TILE_WIDTH + threadIdx.x;

	if (trow < height)
	{
		if ((column - tcolumn) >= 0)
		{
			if ((column - tcolumn - 1) < 0)
			{
				table[(trow*width) + (column - tcolumn)] = createJewel(globalState, (trow * 3 + tcolumn), difficulty);
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

__global__ void pivotBomb(float* table_d, int width, int height, int TILE_WIDTH)
{
	int trow = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int tcolumn = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (trow < height && tcolumn < width) {
		if (!((trow - 1) < 0 || (trow + 1) >= height || (tcolumn - 1) < 0 || (tcolumn + 1) >= width))
		{
			if (trow % 3 == 1 && tcolumn % 3 == 1)
			{
				dim3 dimBlock(3, 3);
				dim3 dimGrid(1, 1);
				
				pivotBombGPU <<<dimGrid, dimBlock >>> (table_d, width, height, trow, tcolumn);
			}
		}
	}
}

int main(int argc, char** argv) {
	int width;
	int height;
	int difficulty;
	char mode;
	int size;
	char savedFile[9] = "save.txt";
	int selection;

	float* table;
	float* table_d;

	curandState* devStates;

	bool playing = true;
	//Configuración inicial del juego
	if (argc == 1)
	{
		std::cout << "width del table: ";
		std::cin >> width;

		std::cout << "height del table: ";
		std::cin >> height;

		std::cout << "Elija difficulty: \n1.-Facil \n2.-Media \n3.-Dificil\n";
		std::cin >> difficulty;

		std::cout << "Automatico?   1.-SI   2.-NO\n";
		std::cin >> selection;
	}
	else
	{
		mode = argv[1][1];
		difficulty = atoi(argv[2]);
		width = atoi(argv[3]);
		height = atoi(argv[4]);

		switch (mode) {
		case 'a': {selection = 1; break; }
		case 'm': {selection = 2; break; }
		default: printf("Valor no valido.\n"); return -1;
		}
	}

	size = width*height;

	//Tamaño de los bloques que se van a emplear.
	int TILE_WIDTH = getTileWidth(width, height);
	if (TILE_WIDTH == -1)
	{
		printf("ERROR: TILE_WIDTH no valido");
		return 0;
	}

	//Creación de las dimensiones de la tesela
	int wdth = ceil(((float)width) / TILE_WIDTH);
	int hght = ceil(((float)height) / TILE_WIDTH);

	//Inicialización del random de CUDA
	cudaMalloc(&devStates, size * sizeof(curandState));
	setup_kernel <<< 1, size >>> (devStates, unsigned(time(NULL)));

	//Creación de punteros al tablero y la versión en GPU del mismo
	table = (float*)malloc(size * sizeof(float));
	cudaMalloc((void**)&table_d, size * sizeof(float));

	//Inicialización del tablero
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);
	rndmTableInit <<<dimGrid, dimBlock >>>(table_d, difficulty, width, height, TILE_WIDTH, devStates);
	cudaMemcpy(table, table_d, size * sizeof(float), cudaMemcpyDeviceToHost);

	//Bucle de juego: se muestran el estado actual y las opciones disponibles, y se procesa la selección del usuario.
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
		/* EXIT */
		case 0: {
			free(table);
			cudaFree(table_d);
			cudaFree(devStates);
			return 0;
		}
		//El jugador desea desplazar una jewel. El juego consulta la jewel a desplazar y la dirección, confirmando que el movimiento
		//seleccionado es válido.
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
				//Intercambio entre las jewels
				swapSpots(table, jewel1X, jewel1Y, direction, width, height, selection, difficulty, TILE_WIDTH, devStates);
			}
			else if (selection == 1)
			{
				//Modo automático activado, se procede a la selección y ejecución de movimiento
				automaticTableAnalysis(difficulty, table, width, height, TILE_WIDTH, devStates);
			}
			break;
		}
		//Opción guardar: el estado de la partida actual se escribe a un fichero externo.
		case 2: {
			save(table, width, height, difficulty, savedFile);
			std::cout << "Guardado completado"<<std::endl;
			break;
		}
		//Opción cargar: se lee el estado de la partida guardada y se recrea
		case 3: {
			//El juego prepara la carga del tablero
			int found = preload(width, height, difficulty, savedFile);
			size = width*height;
			if (found)
			{
				free(table);
				table = (float*)malloc(size * sizeof(float));

				// Carga efectiva del tablero
				load(width, height, table, savedFile);
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
			std::cout << "Elija una bomba:";

			cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);

			/*
				Selección del tipo de bomba y ejecución del mismo. Los tipos disponibles depende del nivel de dificultad seleccionado,
				de acuerdo a las condiciones establecidas en el documento de la práctica.
			*/
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
				std::cout << "Fila: ";
				std::cin >> row;
				dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
				dim3 dimGrid(wdth, hght);
				rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
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
					std::cout << "Fila: ";
					std::cin >> row;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
					break;
				}
				case 2:
				{
					std::cout << "Columna: ";
					std::cin >> column;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					columnBomb <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, TILE_WIDTH, devStates);
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
					std::cout << "Fila: ";
					std::cin >> row;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					rowBomb <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
					break;
				}
				case 2:
				{
					std::cout << "Columna: ";
					std::cin >> column;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					columnBomb <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, TILE_WIDTH, devStates);
					break;
				}
				case 3:
				{
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					pivotBomb <<<dimGrid, dimBlock >>>(table_d, width, height, TILE_WIDTH);
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