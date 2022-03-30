
#include "gpu_kernels.cuh"
#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include "subgraph.cuh"

#include <curand.h>
#include <curand_kernel.h>

__global__ void bfs_kernel(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							unsigned int *value,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = value[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			finalDist = sourceWeight + 1;
			if(finalDist < value[edgeList[i].end])
			{
				atomicMin(&value[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void cc_kernel(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + edgeList[i].w8;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sswp_kernel(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, edgeList[i].w8);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void pr_kernel(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							float *dist,
							float *delta,
							//bool *finished,
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		u_int64_t degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				//*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				u_int64_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				u_int64_t thisto = thisfrom + degree;
				
				for(u_int64_t i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}


__global__ void bfs_async(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sssp_async(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + edgeList[i].w8;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sswp_async(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		
		unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, edgeList[i].w8);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void cc_async(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		u_int64_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		u_int64_t degree = outDegree[id];
		u_int64_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(u_int64_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void pr_async(unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							float *dist,
							float *delta,
							bool *finished,
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		u_int64_t degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				u_int64_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				u_int64_t thisto = thisfrom + degree;
				
				for(u_int64_t i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}

__global__ void init_rand(curandState *randStates, int size) {
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, tId, 0, &randStates[tId]);
}

__device__ u_int32_t uniform_discrete_distribution(curandState &state, u_int32_t n) {
    return ((static_cast<u_int64_t>(curand(&state)) << 32) + curand(&state)) % n;
}

__global__ void rw_kernel(	unsigned int numAllNodes,
							unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
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
		u_int64_t degree = outDegree[id];
		int thisNumWalker = numWalker1[id];

		u_int64_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;

		for (int i = 0; i < thisNumWalker; i++) {
			unsigned int end;
			if (curand_uniform(&randStates[threadIdx.x]) < 0.15) {
				end = uniform_discrete_distribution(randStates[threadIdx.x], numAllNodes);
			}
			else {
				end = edgeList[thisfrom + uniform_discrete_distribution(randStates[threadIdx.x], degree)].end;
			}

			atomicAdd(&numWalker2[end], 1);
		}
	}
}


__global__ void dw_kernel(	unsigned int numAllNodes,
									unsigned int numNodes,
									unsigned int from,
									u_int64_t numPartitionedEdges,
									unsigned int *activeNodes,
									u_int64_t *activeNodesPointer,
									OutEdge *edgeList,
									u_int64_t *outDegree,
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
		u_int64_t degree = outDegree[id];
		int thisNumWalker = numWalker1[id];

		u_int64_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;

		for (int i = 0; i < thisNumWalker; i++) {
			unsigned int end;
			// if (curand_uniform(&randStates[threadIdx.x]) < 0.15) {
			// 	end = uniform_discrete_distribution(randStates[threadIdx.x], numAllNodes);
			// }
			// else {
				end = edgeList[thisfrom + uniform_discrete_distribution(randStates[threadIdx.x], degree)].end;
			// }

			atomicAdd(&numWalker2[end], 1);
		}
	}
}

__global__ void ppr_kernel(	unsigned int numAllNodes,
							unsigned int numNodes,
							unsigned int from,
							u_int64_t numPartitionedEdges,
							unsigned int *activeNodes,
							u_int64_t *activeNodesPointer,
							OutEdge *edgeList,
							u_int64_t *outDegree,
							float *value,
							int *numWalker1,
							int *numWalker2,
							curandState *randStates
							)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(numNodes == 1) 
	{
		unsigned int id = activeNodes[from];
		u_int64_t degree = outDegree[id];
		int thisNumWalker = numWalker1[id] / blockDim.x;
		if (threadIdx.x < numWalker1[id] % blockDim.x) {
			thisNumWalker += 1;
		}

		u_int64_t thisfrom = activeNodesPointer[from]-numPartitionedEdges;

		for (int i = 0; i < thisNumWalker; i++) {
			unsigned int end;
			end = edgeList[thisfrom + uniform_discrete_distribution(randStates[threadIdx.x], degree)].end;

			atomicAdd(&numWalker2[end], 1);
		}
	}
	else if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		u_int64_t degree = outDegree[id];
		int thisNumWalker = numWalker1[id];

		u_int64_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;

		for (int i = 0; i < thisNumWalker; i++) {
			unsigned int end;
			end = edgeList[thisfrom + uniform_discrete_distribution(randStates[threadIdx.x], degree)].end;

			atomicAdd(&numWalker2[end], 1);
		}
	}
}

__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[activeNodes[id+from]] = false;
	}
}

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		int nID = activeNodes[id+from];
		label1[nID] = label1[nID] || label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		label1[nID] = label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveWalkers_pr(unsigned int num_nodes, int *numWalker1, int *numWalker2, float *value)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes) {
		numWalker1[id] = numWalker2[id];
		value[id] += numWalker2[id];
		numWalker2[id] = 0;
	}
}


__global__ void moveWalkers_ppr(unsigned int num_nodes, int *numWalker1, int *numWalker2, float *value, curandState *randStates)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes) {
		numWalker1[id] = 0;
		for (int i = 0; i < numWalker2[id]; i++) {
			if (curand_uniform(&randStates[threadIdx.x]) > 0.15) {
				numWalker1[id]++;
			}
		}

		value[id] += numWalker2[id] - numWalker1[id];
		numWalker2[id] = 0;
	}
}
