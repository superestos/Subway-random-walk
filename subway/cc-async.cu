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
	Stopwatch generateTimer;

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
		graph.value[i] = i;
		graph.label1[i] = true;
		graph.label2[i] = false;
	}


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(u_int64_t), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	generateTimer.start();
	subgen.generate(graph, subgraph);
	generateTimer.stop();

	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	unsigned int gItr = 0;
	
	bool finished;
	bool *d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

	unsigned long totalActiveNodes = 0;
		
	while (subgraph.numActiveNodes>0)
	{
		gItr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);

		cout << "num active nodes: " << subgraph.numActiveNodes << "\n";
		totalActiveNodes += subgraph.numActiveNodes;

		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			copyTimer.start();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			copyTimer.stop();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			
			uint itr = 0;
			do
			{
				itr++;
				finished = true;

				
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				

				computeTimer.start();
				cc_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
														partitioner.fromNode[i],
														partitioner.fromEdge[i],
														subgraph.d_activeNodes,
														subgraph.d_activeNodesPointer,
														subgraph.d_activeEdgeList,
														graph.d_outDegree,
														graph.d_value, 
														d_finished,
														(itr%2==1) ? graph.d_label1 : graph.d_label2,
														(itr%2==1) ? graph.d_label2 : graph.d_label1);

				cudaDeviceSynchronize();
				computeTimer.stop();
				gpuErrorcheck( cudaPeekAtLastError() );
				
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			}while(!(finished));
			
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
		}
		
		generateTimer.start();
		subgen.generate(graph, subgraph);
		generateTimer.stop();
			
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";

	cout << "compute time: " << computeTimer.total() << " ns copy time: " << copyTimer.total() << " ns\n";
	cout << "generate subgraph time: " << generateTimer.total() << " ns\n";

	cout << "total active nodes: " << totalActiveNodes << "\n";
	
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));
			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

