#include "../shared/globals.hpp"


bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

int main(int argc, char** argv)
{
	if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}
	
	string input = string(argv[1]);
	
	if(GetFileExtension(input) == "el")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		vector<Edge> edges;
		Edge newEdge;
		while(getline( infile, line ))
		{
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> newEdge.source;
			ss >> newEdge.end;
			
			edges.push_back(newEdge);
			edgeCounter++;
			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;				
		}			
		infile.close();
		
		uint num_nodes = max + 1;
		u_int64_t num_edges = edgeCounter;
		u_int64_t *nodePointer = new u_int64_t[num_nodes+1];
		OutEdge *edgeList = new OutEdge[num_edges];
		uint *degree = new uint[num_nodes+1];
		for(uint i=0; i<num_nodes; i++)
			degree[i] = 0;
		for(u_int64_t i=0; i<num_edges; i++)
			degree[edges[i].source]++;
		
		u_int64_t counter=0;
		for(uint i=0; i<=num_nodes; i++)
		{
			nodePointer[i] = counter;
			counter = counter + degree[i];
		}
		u_int64_t *outDegreeCounter  = new u_int64_t[num_nodes];
		u_int64_t location;  
		for(uint i=0; i<num_edges; i++)
		{
			location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			outDegreeCounter[edges[i].source]++;  
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
		
		std::ofstream outfile(input.substr(0, input.length()-2)+"bcsr", std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(unsigned int));
		outfile.write((char*)&num_edges, sizeof(u_int64_t));
		outfile.write ((char*)nodePointer, sizeof(u_int64_t)*(num_nodes+1));
		outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
		
		outfile.close();
	}
	else
	{
		cout << "\nInput file format is not supported.\n";
		exit(0);
	}

}
