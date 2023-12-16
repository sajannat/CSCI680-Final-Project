#include <pugixml.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <cuda_runtime.h>

// Define the Node struct
struct Node {
    double latitude;
    double longitude;
};

// Define the Graph type
using Graph = std::unordered_map<int, std::vector<int>>;

// Define the AStarNode struct
struct AStarNode {
    int id;
    double gScore;
    double fScore;
};

// Comparison operator for priority queue
struct CompareAStarNode {
    bool operator()(const AStarNode& a, const AStarNode& b) const {
        return a.fScore > b.fScore;
    }
};

// CUDA kernel for parallel heuristic computation
__global__ void computeHeuristics(Node* nodesA, Node* nodesB, double* results, int numPairs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numPairs) {
        double dx = nodesA[index].latitude - nodesB[index].latitude;
        double dy = nodesA[index].longitude - nodesB[index].longitude;
        results[index] = sqrt(dx * dx + dy * dy);
    }
}

void calculateHeuristicsOnGPU(const Node& currentNode, const std::vector<Node>& neighborNodes, std::vector<double>& heuristics) {
    Node* d_currentNode;
    Node* d_neighborNodes;
    double* d_heuristics;
    int numPairs = neighborNodes.size();

    cudaMalloc(&d_currentNode, sizeof(Node));
    cudaMalloc(&d_neighborNodes, numPairs * sizeof(Node));
    cudaMalloc(&d_heuristics, numPairs * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_currentNode, &currentNode, sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborNodes, neighborNodes.data(), numPairs * sizeof(Node), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numPairs + blockSize - 1) / blockSize;
    computeHeuristics<<<gridSize, blockSize>>>(d_currentNode, d_neighborNodes, d_heuristics, numPairs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
        
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Execution time: " << milliseconds << " milliseconds" << std::endl;
    

    heuristics.resize(numPairs);
    cudaMemcpy(heuristics.data(), d_heuristics, numPairs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_currentNode);
    cudaFree(d_neighborNodes);
    cudaFree(d_heuristics);
}

// Heuristic function (Euclidean distance)
double heuristic(const Node& a, const Node& b) {
    double dx = a.latitude - b.latitude;
    double dy = a.longitude - b.longitude;
    return std::sqrt(dx * dx + dy * dy);
}

double edgeWeight(const Node& a, const Node& b) {
    return heuristic(a, b);
}

void parseOSM(const char* filename, std::unordered_map<int, Node>& nodes, Graph& graph) {
    pugi::xml_document doc;
    if (!doc.load_file(filename)) {
        std::cerr << "Error loading OSM file\n";
        return;
    }

    // Iterate over nodes
    for (pugi::xml_node nodeElem = doc.child("osm").child("node"); nodeElem; nodeElem = nodeElem.next_sibling("node")) {
        int id = nodeElem.attribute("id").as_int();
        double lat = nodeElem.attribute("lat").as_double();
        double lon = nodeElem.attribute("lon").as_double();
        nodes[id] = {lat, lon};
    }

    // Iterate over ways
    for (pugi::xml_node wayElem = doc.child("osm").child("way"); wayElem; wayElem = wayElem.next_sibling("way")) {
        std::vector<int> wayNodes;
        for (pugi::xml_node ndElem = wayElem.child("nd"); ndElem; ndElem = ndElem.next_sibling("nd")) {
            int ref = ndElem.attribute("ref").as_int();
            wayNodes.push_back(ref);
        }

        // Add edges to the graph
        for (size_t i = 0; i < wayNodes.size() - 1; ++i) {
            graph[wayNodes[i]].push_back(wayNodes[i + 1]);
            // If the graph is undirected, add the reverse edge
            // graph[wayNodes[i + 1]].push_back(wayNodes[i]);
        }
    }

    // Add print statements for debugging
    std::cout << "Nodes loaded: " << nodes.size() << '\n';
    std::cout << "Edges loaded: " << graph.size() << '\n';
}

std::vector<int> aStarSearch(const Graph& graph, const std::unordered_map<int, Node>& nodes, int startId, int goalId) {
    std::priority_queue<AStarNode, std::vector<AStarNode>, CompareAStarNode> openSet;
    std::unordered_map<int, double> gScore;
    std::unordered_map<int, int> cameFrom;
    std::unordered_set<int> closedSet;

    AStarNode startNode{startId, 0, heuristic(nodes.at(startId), nodes.at(goalId))};
    openSet.push(startNode);
    gScore[startId] = 0;

    while (!openSet.empty()) {
        AStarNode current = openSet.top();
        openSet.pop();

        if (current.id == goalId) {
            std::vector<int> path;
            for (int nodeId = goalId; nodeId != startId; nodeId = cameFrom[nodeId]) {
                path.push_back(nodeId);
            }
            path.push_back(startId);
            std::reverse(path.begin(), path.end());
            return path;
        }

        if (closedSet.find(current.id) != closedSet.end()) {
            continue;
        }
        closedSet.insert(current.id);

        auto graphIter = graph.find(current.id);
        if (graphIter == graph.end()) {
            continue;
        }

        std::vector<Node> batchNodes;
        for (int neighborId : graphIter->second) {
            if (closedSet.find(neighborId) == closedSet.end()) {
                batchNodes.push_back(nodes.at(neighborId));
            }
        }

        std::vector<double> heuristics;
        calculateHeuristicsOnGPU(nodes.at(current.id), batchNodes, heuristics);

        for (size_t i = 0; i < batchNodes.size(); ++i) {
            int neighborId = graphIter->second[i];
            double tentativeGScore = gScore[current.id] + edgeWeight(nodes.at(current.id), batchNodes[i]);

            if (gScore.find(neighborId) == gScore.end() || tentativeGScore < gScore[neighborId]) {
                gScore[neighborId] = tentativeGScore;
                cameFrom[neighborId] = current.id;

                AStarNode neighborNode{neighborId, tentativeGScore, tentativeGScore + heuristics[i]};
                openSet.push(neighborNode);
            }
        }
    }

    return std::vector<int>(); // No path found
}

int main() {
    std::unordered_map<int, Node> nodes;
    Graph graph;

    parseOSM("map1.osm", nodes, graph);

    int startNodeId = 504338530;
    // long long goalNodeId = 9598507043;
    int goalNodeId = 504348423;

    // Running the A* search algorithm
    // auto startTime = std::chrono::high_resolution_clock::now();

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    std::vector<int> path = aStarSearch(graph, nodes, startNodeId, goalNodeId);
    // auto endTime = std::chrono::high_resolution_clock::now();
    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float milliseconds = 0;

    // Output the path
    if (!path.empty()) {
        std::cout << "Path found: ";
        for (int nodeId : path) {
            std::cout << nodeId << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "No path found." << std::endl;
    }

    // Output the execution time
    // std::chrono::duration<double, std::milli> executionTime = endTime - startTime;
    // std::cout << "Execution time: " << executionTime.count() << " milliseconds" << std::endl;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Execution time: " << milliseconds << " milliseconds" << std::endl;

    return 0;
}