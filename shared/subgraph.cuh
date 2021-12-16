#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP


#include "globals.hpp"


template <class E>
class Subgraph
{
private:

public:
	uint num_nodes;
	u_int64_t num_edges;
	uint numActiveNodes;
	
	uint *activeNodes;
	u_int64_t *activeNodesPointer;
	E *activeEdgeList;
	
	uint *d_activeNodes;
	u_int64_t *d_activeNodesPointer;
	E *d_activeEdgeList;
	
	ull max_partition_size;
	
	Subgraph(uint num_nodes, u_int64_t num_edges);
};

#endif	//	SUBGRAPH_HPP



