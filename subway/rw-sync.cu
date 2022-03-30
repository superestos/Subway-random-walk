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



int main(int argc, char** argv)
{	
	Stopwatch copyTimer;
	Stopwatch computeTimer;
	Stopwatch generateTimer;

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

	std::random_device rd;
	std::mt19937 rng{rd()}; 
	std::uniform_int_distribution<int> uniform(0, graph.num_nodes - 1);
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = 0;
		numWalker1[i] = 0;
	}

	u_int32_t numWalker = graph.num_nodes * 2;
	for(u_int32_t i = 0; i < numWalker; i++) {
		numWalker1[uniform(rng)]++;
	}

	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(u_int64_t), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_numWalker1, numWalker1, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	generateTimer.start();
	subgen.generate(graph, subgraph, d_numWalker1);
	generateTimer.stop();

	cout << "generate subgraph" << endl;

	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint gItr = 0;

	unsigned long totalActiveNodes = 0;

	u_int32_t *visit_count;
	cudaMalloc(&visit_count, sizeof(u_int32_t) * graph.num_nodes);
	u_int32_t *h_visit_count = new u_int32_t[graph.num_nodes];
		
	for (; gItr < 10; gItr++)
	{
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);

		cout << "num active nodes: " << subgraph.numActiveNodes << "\n";

		u_int64_t activeEdges = 0;
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			activeEdges += partitioner.partitionEdgeSize[i];
		}

		cout << "num active edges: " << activeEdges << "\n";
		
		totalActiveNodes += subgraph.numActiveNodes;

		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			copyTimer.start();
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

		//computeTimer.stop();
		moveWalkers_pr<<<graph.num_nodes/512 + 1, 512>>>(graph.num_nodes, d_numWalker1, d_numWalker2, visit_count);
		//cudaDeviceSynchronize();
		//computeTimer.stop();
		
		generateTimer.start();
		subgen.generate(graph, subgraph, d_numWalker1);
		generateTimer.stop();
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	cout << "Number of iterations = " << gItr << endl;

	cout << "compute time: " << computeTimer.total() << " ns copy time: " << copyTimer.total() << " ns\n";
	cout << "generate subgraph time: " << generateTimer.total() << " ns\n";

	cout << "total active nodes: " << totalActiveNodes << "\n";
	
	gpuErrorcheck(cudaMemcpy(h_visit_count, visit_count, graph.num_nodes*sizeof(u_int32_t), cudaMemcpyDeviceToHost));

	unsigned long sum = 0;
	for (unsigned i = 0; i < graph.num_nodes; i++) {
		sum += h_visit_count[i];
	}
	cout << "sum: " << sum << endl;
	
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));

			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

