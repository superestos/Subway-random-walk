#include "graph.cuh"
#include "gpu_error_check.cuh"

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(u_int64_t));
		
		nodePointer = new u_int64_t[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(u_int64_t)*(num_nodes+1));
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new u_int64_t[num_nodes];
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	value  = new unsigned int[num_nodes];
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(u_int64_t)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;


}

//--------------------------------------

template <class E>
GraphPR<E>::GraphPR(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string GraphPR<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphPR<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void GraphPR<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void GraphPR<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(u_int64_t));
		
		nodePointer = new u_int64_t[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(u_int64_t)*(num_nodes+1));
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new u_int64_t[num_nodes];
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	

	value  = new float[num_nodes];
	delta  = new float[num_nodes];
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(u_int64_t)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
	

}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;

template class GraphPR<OutEdge>;
template class GraphPR<OutEdgeWeighted>;
