#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include <fstream>

//Analiza las properties de la tarjeta grafica para devolver el sizaño adecuado de tile, sizbien trata el sizaño del table
int obtenerTileWidth(int width, int height) {
	float minSize = 0;

	if (width > height) minSize = width;
	else minSize = height;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);

	int maxThreads = properties.maxThreadsPerBlock;

	if (width == height) {	//Si la matriz es cuadrada, para no tener 1 solo bloque
		if (minSize / 32 > 1 && maxThreads == 1024) { //Solo si tiene 1024 hilos por bloque podra ser de 32x32
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
	else {	//si la matriz no es cuadrada
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

/* Funcion para inicializar la matriz de gemas */
__global__ void randomTableInit(float *table, int difficulty, int width, int height, int TILE_WIDTH, curandState* globalState) {
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

/*Recibe las coordenadas de las jewels a eliminar y mueve las rows que tiene que bajar a partir de ellas, emplea
una copia del table para evitar race conditions*/
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
/*Funcion que prepara y llama el kernel con su mismo nombre, genera todos los datos necesarios*/
void eraseJewels(float* table, float* erasedJewels, int difficulty, int width, int height, int TILE_WIDTH, curandState* globalState) {
	float *table_d;
	float *erasedJewels_d;
	float *auxTable_d;
	int size = width * height * sizeof(float);
	
	int max = 0;

	//Para saber que medida es la más grande, ya que no se pueden eliminar más jewels seguidas que esa medida
	if (height >= width) max = height;
	else max = width;

	//table a GPU y la copia del table
	cudaMalloc((void**)&table_d, size);
	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&auxTable_d, size);
	cudaMemcpy(auxTable_d, table, size, cudaMemcpyHostToDevice);

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

	//En caso de que este complesizente escrito
	if (!altered) end = max * 2;

	//Cantidad de bloques de wdtho de medida TILE_WIDTH
	int wdth = ceil(((double)width) / TILE_WIDTH);

	//Cantidad de bloques de hghto con medida TILE_WIDTH
	int hght = ceil(((double)height) / TILE_WIDTH);

	//Configuracion de ejecucion
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);
	eraseJewelsKernel <<<dimGrid, dimBlock>>> (table_d, auxTable_d, erasedJewels_d, difficulty, width, height, end, TILE_WIDTH, globalState);

	//Se recupera el table actualizado
	cudaMemcpy(table, table_d, size, cudaMemcpyDeviceToHost);

	//Libera memoria
	cudaFree(table_d);
	cudaFree(erasedJewels_d);
	cudaFree(auxTable_d);
}
//Analiza el movimiento manual, usando las coordenadas de la nueva posicion de la jewel selectionada
void manualTableAnalysis(int difficulty, float* table, int width, int height, int x, int y, int TILE_WIDTH, curandState* globalState) {
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
	
	int potentialJewelsLeft = 0;
	int potentialJewelsRight = 0;

	//Si tiene por la izquierda
	if ((x - 1 + y*width >= 0) && table[x - 1 + y*width] == table[x + y*width]) {
		int i = 1;
		while ((x - i + y*width >= 0) && (x - i >= 0) && table[x - i + y*width] == table[x + y*width]) {
			potentialJewelsLeft++;
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
	if (1 + potentialJewelsLeft + potentialJewelsRight >= 3) {
		int shghto = 0;
		
		for (int j = potentialJewelsLeft; j >= (1); j--) {
			erasedJewels[shghto] = x - j;
			erasedJewels[shghto + 1] = y;
			shghto += 2;
		}
		
		erasedJewels[potentialJewelsLeft * 2] = x;
		erasedJewels[potentialJewelsLeft * 2 + 1] = y;
		
		shghto = 2;
		for (int k = 1; k <= potentialJewelsRight; k++) {
			erasedJewels[shghto + potentialJewelsLeft * 2] = x + k;
			erasedJewels[shghto + potentialJewelsLeft * 2 + 1] = y;
			shghto += 2;
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

			int shghto = 0;
			for (int j = potentialJewelsBelow; j >= (1); j--) {
				erasedJewels[shghto] = x;
				erasedJewels[shghto + 1] = y - j;
				shghto += 2;
			}
			
			erasedJewels[potentialJewelsBelow * 2] = x;
			erasedJewels[potentialJewelsBelow * 2 + 1] = y;

			shghto = 2;
			for (int k = 1; k <= potentialJewelsAbove; k++) {
				erasedJewels[shghto + potentialJewelsBelow * 2] = x;
				erasedJewels[shghto + potentialJewelsBelow * 2 + 1] = y + k;
				shghto += 2;
			}
		}
	}
	
	//Las elimina
	eraseJewels(table, erasedJewels, difficulty, width, height, TILE_WIDTH, globalState);
	
	free(erasedJewels);
}
//Intercambia la jewel selectionadas con la jewel en la dirección indicada
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

	//Analiza el movimiento para ver si se pueden eliminar jewels
	manualTableAnalysis(difficulty, table, width, height, jewel2X, jewel2Y,TILE_WIDTH, globalState);
}

/*Escribe en un table auxiliar la cantidad de jewels que se eliminarian moviendo una jewel (x,y) hacia la derecha
paralelizable ya que todos los hilos (cada hilo 1 jewel) tienen que expandirse hacia la derecha para ver hasta donde llegarian a eliminar*/
__global__ void automaticTableAnalysisKernel(float *table_d, float *aux_d, int difficulty, int width, int height, int TILE_WIDTH, curandState* globalState) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	//Posicion real dentro del table
	tx += block_x * TILE_WIDTH;
	ty += block_y * TILE_WIDTH;

	//Array dinamico en memoria compartida, velocidad de accesoo mucho mayor que con global
	extern __shared__ float sharedTable[];

	//Entre todos los hilos, rellenan por completo el auxiliar en memoria compartida
	sharedTable[tx + ty*width] = aux_d[tx + ty*width];

	//Esperan a que todos los hilos copien el valor, creando un table auxiliar completo en compartida.
	__syncthreads();

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

			sharedTable[tx + ty*width] = potentialJewelsRight + 1;
		}
		else {
			sharedTable[tx + ty*width] = 1;
		}
	}
	else {
		sharedTable[tx + ty*width] = 1;
	}

	//Se esperan a que todos hayan calculado para actualizar la matriz a devolver
	__syncthreads();

	aux_d[tx + ty*width] = sharedTable[tx + ty*width];
}


//Analiza la mejor opcion y la ejecuta en funcion de lo que devuelve el kernel
void automaticTableAnalysis(int difficulty, float* table, int width, int height, int TILE_WIDTH, curandState* globalState) {
	float *table_d;
	float *aux_d;
	float *aux;
	
	//sizaño del table para asignar memoria
	int size = width * height * sizeof(float);
	int size1 = width * height;
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
	for (int p = 0; p < size1; p++) {
		aux[p] = 1;
	}

	//table a GPU
	cudaMalloc((void**)&table_d, size);

	cudaMemcpy(table_d, table, size, cudaMemcpyHostToDevice);
	//Auxiliar de conteo a GPU

	cudaMalloc((void**)&aux_d, size);

	cudaMemcpy(aux_d, aux, size, cudaMemcpyHostToDevice);

	//Cantidad de bloques de wdtho de medida TILE_WIDTH
	int wdth = ceil(((double)width) / TILE_WIDTH);

	//Cantidad de bloques de hghto con medida TILE_WIDTH
	int hght = ceil(((double)height) / TILE_WIDTH);

	//Configuracion de ejecucion
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);

	//Inicio del kernel

	automaticTableAnalysisKernel <<<dimGrid, dimBlock, 2*width * height * sizeof(float) >>> (table_d, aux_d, difficulty, width, height, TILE_WIDTH, globalState);

	//Transfiere el resultado de la GPU al host
	cudaMemcpy(aux, aux_d, size, cudaMemcpyDeviceToHost);

	int bestX = 0;
	int bestY = 0;
	int bestValue = 0;

	//Se busca el movimiento con el mayor numero de jewels eliminadas
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (aux[x + y*width] > bestValue) {
				bestValue = aux[x + y*width];
				bestX = x;
				bestY = y;
			}
		}
	}

	//Si se pueden eliminar se ejecuta el movimiento, con lo que ello conlleva
	if (bestValue >= 3) {
		swapSpots(table, bestX, bestY, 4, width, height, 1, difficulty, TILE_WIDTH, globalState);
	}
	free(aux);
	free(erasedJewels);
	cudaFree(table_d);
	cudaFree(aux_d);
}

bool preload(int& width, int& height, int& difficulty, char* file)
{
	std::ifstream loadF(file);
	char siz[4];
	if (!loadF.is_open())
	{
		std::cout << "ERROR: no existe un archivo save." << std::endl;
		return false;
	}

	loadF.getline(siz, 4);

	width = (int)siz[0] - 48;
	height = (int)siz[1] - 48;
	difficulty = (int)siz[2] - 48;

	loadF.close();
	return true;
}
void load(int width, int height, float*  table, char* file)
{
	char* array = (char*)malloc(width*height + 1 + 3);
	std::ifstream loadF(file);
	loadF.getline(array, (width*height + 1 + 3));
	for (int i = 0; i < width*height; i++)
	{
		table[i] = array[i + 3] - 48;
	}
	free(array);
	loadF.close();
}

void save(float* table, int width, int height, int difficulty, char* file)
{
	//Sistema de save
	std::ofstream savedFile;
	savedFile.open(file);
	savedFile.clear();
	/* Almacenar width y height*/
	savedFile << width;
	savedFile << height;
	savedFile << difficulty;
	/* Almacenar Resto */
	for (int index = 0; index < width*height; index++)
	{
		savedFile << table[index];
	}
	savedFile.close();
}
/* Funcion que elimina una row */
__global__ void bombRow(float* table, int width, int height, int difficulty, int row, int TILE_WIDTH, curandState* globalState) {

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

/* Funcion que elimina una column */
__global__ void bombColumn(float* table, int width, int height, int difficulty, int column, int TILE_WIDTH, curandState* globalState) {

	int trow = blockIdx.y*TILE_WIDTH + threadIdx.y;
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
	__shared__ int aux[9];
	int trow = threadIdx.y;
	int tcolumn = threadIdx.x;

	if (trow < 3)
	{
		if (tcolumn < 3)
		{
			/* Memoria compartida */
			aux[trow + tcolumn * 3] = table[((row + 1) - trow)*width + ((column + 1) - tcolumn)];
			__syncthreads();
			table[((row + 1) - trow)*width + ((column - 1) + tcolumn)] = aux[trow * 3 + tcolumn];
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
	//Matriz de sizaño variable de floats, un array de height*width
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
	/* Valores por argumento*/
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

	//sizaño de los bloques a crear en CUDA
	int TILE_WIDTH = obtenerTileWidth(width, height);
	if (TILE_WIDTH == -1)
	{
		printf("ERROR: TILE_WIDTH no valido");
		return 0;
	}

	//Cantidad de bloques de wdtho de medida TILE_WIDTH
	int wdth = ceil(((float)width) / TILE_WIDTH);

	//Cantidad de bloques de hghto con medida TILE_WIDTH
	int hght = ceil(((float)height) / TILE_WIDTH);

	/* Inicializacion random en CUDA */
	cudaMalloc(&devStates, size * sizeof(curandState));

	/* Creacion de las Seeds */
	setup_kernel <<< 1, size >>> (devStates, unsigned(time(NULL)));

	/* Reservar memoria para table y table_d */
	table = (float*)malloc(size * sizeof(float));
	cudaMalloc((void**)&table_d, size * sizeof(float));

	/* Se inicializa la matriz */
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(wdth, hght);
	randomTableInit <<<dimGrid, dimBlock >>>(table_d, difficulty, width, height, TILE_WIDTH, devStates);
	cudaMemcpy(table, table_d, size * sizeof(float), cudaMemcpyDeviceToHost);

	//Bucle principal del juego
	while (playing) {

		printTable(table, width, height);

		int jewel1X = 0;
		int jewel1Y = 0;
		int command = 0;

		std::cout << "Acción a realizar:\n";
		std::cout << "(1) Intercambiar Jewels\n";
		std::cout << "(2) Guardar partida\n";
		std::cout << "(3) load partida\n";
		std::cout << "(9) Usar una bomb\n";
		std::cout << "(0) Exit\n";
		std::cout << "Elija command: ";

		std::cin >> command;

		switch (command) {
			/* EXIT */
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
				std::cout << "direction a seguir para intercambio de posiciones: \n 1.-Arriba\n 2.-Abajo\n 3.-Izquierda\n 4.-Derecha\n";
				std::cin >> direction;

				if (direction > 4 && direction > 1) {
					printf("direction erronea.\n");
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
				swapSpots(table, jewel1X, jewel1Y, direction, width, height, selection, difficulty, TILE_WIDTH, devStates);

			}
			else if (selection == 1)
			{
				/* Analisis automatico */
				automaticTableAnalysis(difficulty, table, width, height, TILE_WIDTH, devStates);
			}
			break;
		}
				/* Guardar Partida */
		case 2: {

			save(table, width, height, difficulty, savedFile);
			std::cout << "save correcto.\n";
			break;
		}
				/* load Partida */
		case 3: {

			/* Precarga de table */
			bool found = preload(width, height, difficulty, savedFile);

			if (found)
			{
				/* load table */
				load(width, height, table, savedFile);
				std::cout << "Se ha cargado el table: \n";
			}
			else {
				std::cout << "No existe ninguna partida guardada.\n";
			}
			break;

		}
				/* bombs */
		case 9: {

			int bomb = 0;
			int row = 0; int column = 0;
			std::cout << "Elija una bomb:";

			cudaMemcpy(table_d, table, size * sizeof(float), cudaMemcpyHostToDevice);

			/* bombs por tipo de difficulty */
			switch (difficulty) {
			case 1: {
				std::cout << "(1) bomb de row ";
				std::cout << "\nEleccion: ";
				std::cin >> bomb;

				if (bomb != 1)
				{
					printf("bomb erronea.\n");
					continue;
				}
				std::cout << "row: ";
				std::cin >> row;
				dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
				dim3 dimGrid(wdth, hght);
				bombRow <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
				break;
			}
			case 2: {
				std::cout << "(1) bomb de row";
				std::cout << "(2) bomb de column";
				std::cout << "\nEleccion: ";
				std::cin >> bomb;

				if (bomb < 1 && bomb > 2)
				{
					printf("bomb erronea.\n");
					continue;
				}
				switch (bomb) {
				case 1:
				{
					std::cout << "row: ";
					std::cin >> row;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					bombRow <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
					break;
				}
				case 2:
				{
					std::cout << "column: ";
					std::cin >> column;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					bombColumn <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, TILE_WIDTH, devStates);
					break;
				}
				}
				break;
			}
			case 3: {
				std::cout << "(1) bomb de row";
				std::cout << "(2) bomb de column";
				std::cout << "(3) bomb de rotacion 3x3";
				std::cout << "\nEleccion: ";
				std::cin >> bomb;

				if (bomb < 1 && bomb > 3)
				{
					printf("bomb erronea.\n");
					continue;
				}
				switch (bomb) {
				case 1:
				{
					std::cout << "row: ";
					std::cin >> row;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					bombRow <<<dimGrid, dimBlock >>> (table_d, width, height, difficulty, row, TILE_WIDTH, devStates);
					break;
				}
				case 2:
				{
					std::cout << "column: ";
					std::cin >> column;
					dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
					dim3 dimGrid(wdth, hght);
					bombColumn <<<dimGrid, dimBlock >>>(table_d, width, height, difficulty, column, TILE_WIDTH, devStates);
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