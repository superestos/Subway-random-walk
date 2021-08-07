#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"
#include "../shared/test.cuh"
#include "../shared/test.cu"
#include "../shared/stopwatch.h"

#include <curand.h>
#include <curand_kernel.h>

__global__ void init_rand(curandState *randStates, int size) {
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, tId, 0, &randStates[tId]);
}

__global__ void rw_kernel(	unsigned int numAllNodes,
							unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *value,
							int *numWalker1,
							int *numWalker2,
							curandState *randStates
							)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		int thisNumWalker = numWalker1[id];

		unsigned int thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;

		for (int i = 0; i < thisNumWalker; i++) {
			unsigned int end;
			if (degree == 0 || curand_uniform(&randStates[threadIdx.x]) < 0.15) {
				end = (unsigned int)(numAllNodes * curand_uniform(&randStates[threadIdx.x]));
			}
			else {
				end = edgeList[thisfrom + (unsigned int)(degree * curand_uniform(&randStates[threadIdx.x]))].end;
			}
			end = (end >= numAllNodes)? numAllNodes - 1: end;

			atomicAdd(&numWalker2[end], 1);
		}
	}
}

/*
__global__ void moveWalkers(unsigned int * activeNodes, int *numWalker1, int *numWalker2, float *value, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		numWalker1[nID] = numWalker2[nID];
		value[nID] += (numWalker2[nID] + 0.0) / 10.0;
		numWalker2[nID] = 0;
	}
}
*/

__global__ void moveWalkers(unsigned int num_nodes, int *numWalker1, int *numWalker2, float *value)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes) {
		numWalker1[id] = numWalker2[id];
		value[id] += (numWalker2[id] + 0.0) / 10.0;
		numWalker2[id] = 0;
	}
}

int main(int argc, char** argv)
{	
	Stopwatch copyTimer;
	Stopwatch computeTimer;

	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	GraphPR<OutEdge> graph(arguments.input, true);
	graph.ReadGraph();
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	int *numWalker1 = new int[graph.num_nodes];
	int *d_numWalker1, *d_numWalker2;

	cudaMalloc(&d_numWalker1, sizeof(int) * graph.num_nodes);
	cudaMalloc(&d_numWalker2, sizeof(int) * graph.num_nodes);

	curandState *randStates;
	cudaMalloc(&randStates, sizeof(curandState) * 512);
	init_rand<<<1, 512>>>(randStates, 512);
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = 0;
		numWalker1[i] = 1;
	}


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_numWalker1, numWalker1, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	subgen.generate(graph, subgraph, d_numWalker1);

	cout << "generate subgraph" << endl;

	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint gItr = 0;

	unsigned long totalActiveNodes = 0;
		
	for (; gItr < 10; gItr++)
	{
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);

		cout << "num active nodes: " << subgraph.numActiveNodes << "\n";
		totalActiveNodes += subgraph.numActiveNodes;

		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			copyTimer.start();
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			copyTimer.stop();
			
			computeTimer.start();
			rw_kernel<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(graph.num_nodes,
												partitioner.partitionNodeSize[i],
												partitioner.fromNode[i],
												partitioner.fromEdge[i],
												subgraph.d_activeNodes,
												subgraph.d_activeNodesPointer,
												subgraph.d_activeEdgeList,
												graph.d_outDegree,
												graph.d_value,
												d_numWalker1,
												d_numWalker2,
												randStates);		

			//moveWalkers<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, d_numWalker1, d_numWalker2, graph.d_value, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			cudaDeviceSynchronize();
			computeTimer.stop();
			gpuErrorcheck( cudaPeekAtLastError() );	
	
		}

		moveWalkers<<<graph.num_nodes/512 + 1, 512>>>(graph.num_nodes, d_numWalker1, d_numWalker2, graph.d_value);
		
		copyTimer.start();
		subgen.generate(graph, subgraph, d_numWalker1);
		copyTimer.stop();
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	cout << "Number of iterations = " << gItr << endl;

	cout << "compute time: " << computeTimer.total() << " ns copy time: " << copyTimer.total() << " ns\n";

	cout << "total active nodes: " << totalActiveNodes << "\n";
	
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(float), cudaMemcpyDeviceToHost));

	unsigned long sum = 0;
	for (unsigned i = 0; i < graph.num_nodes; i++) {
		sum += std::lround(graph.value[i] * 10);
	}
	cout << "sum: " << sum << endl;
	
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));

			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

