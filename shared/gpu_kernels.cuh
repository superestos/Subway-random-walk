#include <curand.h>
#include <curand_kernel.h>

#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"

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
							bool *label2);

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
							bool *label2);

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
							bool *label2);
							
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
							bool *label2);
							
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
							float acc);						

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
							bool *label2);	
							
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
							bool *label2);
							
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
							bool *label2);
							
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
							bool *label2);		
							
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
							float acc);	

__global__ void init_rand(curandState *randStates, int size);

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
							);

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
							);

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
							);

__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from);

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

__global__ void moveWalkers_pr(unsigned int num_nodes, int *numWalker1, int *numWalker2, float *value);

__global__ void moveWalkers_ppr(unsigned int num_nodes, int *numWalker1, int *numWalker2, float *value, curandState *randStates);
