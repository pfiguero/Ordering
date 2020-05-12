#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>

#include <map>
#include <chrono>

#include <math.h>

#include <iostream>
using namespace std;


// Defines from https://gist.github.com/Tener/803377/38562ed70bd627dac09946222d1005d7d4e95e50
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      exit( EXIT_FAILURE );}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      exit( EXIT_FAILURE );}} while(0)


// deprecatd... Just to try my own generator, but there are way better ones
__global__
void gpuRandom(int* x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//	int stride = blockDim.x * gridDim.x;
	x[index] = 17 * index % 23;
}


class Solver
{
protected:
	long seed;
	long numElems;
	virtual ostream& print(ostream& os) = 0;
public:
	Solver(long n, long s) { numElems = n; seed = s;  }
	inline virtual ~Solver() {};
	virtual void createUnordered() = 0;
	virtual void order() = 0;
	inline virtual void deleteData() = 0;
	inline friend ostream& operator<<(ostream& os, Solver& dt)
	{
		if (dt.numElems > 1000)
		{
			os << "Way too many to show" << std::endl;
			return os;
		}
		else
		{
			return dt.print(os);
		}
	};
};

template<typename T>
class CPUSolver: public Solver
{
	T* array;
protected:
	virtual ostream& print(ostream& os);
public:
	inline CPUSolver<T>(long n, long s = 129229): Solver(n,s) { array = new T[numElems];  }
	inline virtual ~CPUSolver() { };
	inline virtual void deleteData() { delete array; };
	virtual void createUnordered();
	virtual void order();
};

template<typename T>
void CPUSolver<T>::createUnordered()
{
	// The formula might break with too big numbers...
	for (long i = 0; i < numElems; i++)
	{
		array[i] = (T) rand();
	}
}

int cmpfunc(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

// ONLY WORKS WITH INT!!!
template<typename T>
void CPUSolver<T>::order()
{
	qsort(array, numElems, sizeof(int), cmpfunc);
}

template<typename T>
ostream& CPUSolver<T>::print(ostream& os)
{
	for (long i = 0; i < numElems; i++)
	{
		os << array[i] << ", ";
	}
	os << std::endl;
	return os;
}


template<typename T>
class GPUSolver : public Solver
{
	T* x;
protected:
	virtual ostream& print(ostream& os);
public:
	inline GPUSolver<T>(long n, long s = 129229) : Solver(n, s) {
		CUDA_CALL(cudaMallocManaged(&x, numElems * sizeof(T)));
	}
	// Can't free in the destructor, since there are objects created with a copy constructor
	// that can't free the info in this way... :(
	inline virtual   ~GPUSolver<T>() { };
	inline virtual void deleteData() 
	{
		if (x != NULL) { CUDA_CALL(cudaFree(x)); x = NULL; }
	};
	virtual void createUnordered();
	virtual void order();
};

void bitonic_sort(float* values, long numElems);
void bitonic_sort(int* values, long numElems);

template<typename T>
void GPUSolver<T>::createUnordered()
{
	//int blockSize = 256;
	//int numBlocks = (numElems + blockSize - 1) / blockSize;
	//gpuRandom << <numBlocks, blockSize >> > ( x );

	//// Wait for GPU to finish before accessing on host
	//cudaDeviceSynchronize();

	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	/* Generate n floats on device */
	CURAND_CALL(curandGenerate(gen, (unsigned int*)x, numElems));

	// Wait for GPU to finish before accessing on host
	CUDA_CALL(cudaDeviceSynchronize());
	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gen));
}

// ONLY WORKS WITH FLOAT!!!
template<typename T>
void GPUSolver<T>::order()
{
	bitonic_sort(x,numElems);
}


template<typename T>
ostream& GPUSolver<T>::print(ostream& os)
{
	for (long i = 0; i < numElems; i++)
	{
		os << x[i] << ", ";
	}
	os << std::endl;
	return os;
}


// Services 

// Create a new set of numbers of type T. It returns the id of such set
// ONLY TESTED with int!!!
template<typename T>
long createSet(bool isGpu, long nElems, long seed = 129229);

// returns the ids of all sets available

// returns the metadata of the set of the given id

// returns a subset of the set. The parameters indicate which part

// order the given set
void orderSet(long id);

// deletes everything
void deleteAll();


struct MetadataSolver {
	long id;
	long numElems;
	Solver* solver;
	double nanosCreate;
	double nanosOrder;
	MetadataSolver() { id = -1; numElems = 0;  solver = NULL; nanosCreate = -1; }
	inline void deleteSolverData() 
	{
		if (solver != NULL) { solver->deleteData(); solver = NULL; }
	}
	MetadataSolver(long i, long n, Solver* s, double d) { id = i; numElems = n;  solver = s; nanosCreate = d; }
	MetadataSolver(const MetadataSolver& m) { id = m.id; numElems = m.numElems;  solver = m.solver; nanosCreate = m.nanosCreate; }
} ;

static long numSetIDs = 0;
typedef map<long, MetadataSolver> TupleMap;
static TupleMap sets;

int main1(int argc, char* argv[])
{
	Solver* s = NULL;
	long id;

	// just for the GPU profiler to have something to report
	float* tt;
	cudaMallocManaged(&tt, 5 * sizeof(float));

	long numElems = 256 * 256;
	if (argc <= 1)
	{
		id = createSet<int>(false, numElems, 129229);
		s = sets[id].solver;
		orderSet(id);
		//s = new CPUSolver<int>(100, 129229);
	}
	else
	{
		id = createSet<int>(true, numElems, 129229);
		s = sets[id].solver;
		orderSet(id);
		//s = new GPUSolver<int>(100, 129229);
	}
	cout << (*s);
	return 0;
}

int mainInt(int argc, char* argv[])
{
	long numElems = 256;
	int exponente;

	const int EXIT = 10;
	int opcion = EXIT;

	long id;

	// just for the GPU profiler to have something to report
	float* tt;
	cudaMallocManaged(&tt, 5 * sizeof(float));

	do{
		cout << "Escoja una opción: " << endl;
		cout << "  1. Crear y ordenar conjunto en CPU " << endl;
		cout << "  2. Crear y ordenar conjunto en GPU " << endl;
		cout << "  3. Borrar todo" << endl;
		cout << "  " << EXIT << ". Salir " << endl;
		cin >> opcion;
		if (opcion == EXIT)
		{
			break;
		}
		else if (opcion == 3)
		{
			deleteAll();
			cout << "Todo Borrado" << endl;
		}
		else if( opcion == 1 || opcion == 2)
		{
			cout << "exponente de 256?" << endl;
			cin >> exponente;
			numElems = 256 * exponente;

			if (opcion == 1 )
			{
				id = createSet<int>(false, numElems, 129229);
				orderSet(id);
			}
			else if (opcion == 2 )
			{
				id = createSet<int>(true, numElems, 129229);
				orderSet(id);
			}
		}
	} while (opcion != EXIT);

	return 0;
}

int main(int argc, char* argv[])
{
	return mainInt(argc, argv);
}

template<typename T>
long createSet(bool isGpu, long nElems, long seed)
{
	Solver* s = NULL;
	long id = ++numSetIDs;
	auto start = std::chrono::high_resolution_clock::now();
	if (isGpu)
	{
		s = new GPUSolver<T>(nElems, seed);
	}
	else
	{
		s = new CPUSolver<T>(nElems, seed);
	}
	// create the unordered set
	s->createUnordered();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::nano> elapsed = finish - start;
	std::cout << "Elapsed Time numElems " << nElems << " creation: " << fixed << elapsed.count() << " nanoseconds, typeid: " << typeid(elapsed.count()).name() << std::endl;

	sets[id] = MetadataSolver(id, nElems, s, elapsed.count());
	return id;

}

void orderSet(long id)
{
	Solver* s = sets[id].solver;
	if(s!= NULL)
	{
		auto start = std::chrono::high_resolution_clock::now();
		s->order();
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::nano> elapsed = finish - start;
		std::cout << "Elapsed Time numElems " << sets[id].numElems << " order: " << fixed << elapsed.count() << " nanoseconds, typeid: " << typeid(elapsed.count()).name() << std::endl;
		sets[id].nanosOrder = elapsed.count();
	}
}

void deleteAll()
{
	for (auto it = sets.begin(); it != sets.end(); ) {
		(it->second).deleteSolverData();
		it = sets.erase(it);
	}
}
