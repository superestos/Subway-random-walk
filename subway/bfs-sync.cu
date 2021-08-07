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
#include "../shared/stopwatch.h"

int main(int argc, char** argv)
{
	Stopwatch copyTimer;
	Stopwatch computeTimer;

	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	Graph<OutEdge> graph(arguments.input, false);
	graph.ReadGraph();
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = DIST_INFINITY;
		graph.label1[i] = false;
		graph.label2[i] = false;
	}
	graph.value[arguments.sourceNode] = 0;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	copyTimer.start();
	subgen.generate(graph, subgraph);
	copyTimer.stop();

	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint itr = 0;

	unsigned long totalActiveNodes = 0;
		
	while (subgraph.numActiveNodes>0)
	{
		itr++;
		
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
			moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			bfs_kernel<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													graph.d_value, 
													//d_finished,
													graph.d_label1,
													graph.d_label2);

			cudaDeviceSynchronize();
			computeTimer.stop();
			gpuErrorcheck( cudaPeekAtLastError() );	
		}
		
		copyTimer.start();
		subgen.generate(graph, subgraph);
		copyTimer.stop();
	}
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	cout << "Number of iterations = " << itr << endl;

	cout << "compute time: " << computeTimer.total() << " ns copy time: " << copyTimer.total() << " ns\n";

	cout << "total active nodes: " << totalActiveNodes << "\n";
	
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));
			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

