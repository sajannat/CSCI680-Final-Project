#include <pugixml.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <chrono>

struct Node {
    double latitude;
    double longitude;
};

using Graph = std::unordered_map<int, std::vector<int>>;

struct AStarNode {
    int id;
    double gScore;
    double fScore;

    // Comparison operator for priority queue
    bool operator<(const AStarNode& other) const {
        return fScore > other.fScore; // Note: this is reversed because priority_queue is a max heap
    }
};

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
    std::priority_queue<AStarNode, std::vector<AStarNode>> openSet;
    std::unordered_map<int, double> gScore; // Cost from start along the best-known path
    std::unordered_map<int, int> cameFrom;  // Previous node in the optimal path from the source
    std::unordered_set<int> closedSet;      // Set of nodes already evaluated

    // Initialize the starting node
    AStarNode startNode;
    startNode.id = startId;
    startNode.gScore = 0; // Start node gScore is zero

    auto startTime = std::chrono::high_resolution_clock::now();

    startNode.fScore = heuristic(nodes.at(startId), nodes.at(goalId));

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> executionTime = endTime - startTime;

    std::cout << "Execution time: " << executionTime.count() << " milliseconds" << std::endl;

    openSet.push(startNode);
    gScore[startId] = 0;

    while (!openSet.empty()) {
        AStarNode current = openSet.top();
        openSet.pop();

        std::cout << "Processing Node: " << current.id << " with fScore: " << current.fScore << std::endl;

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

        for (int neighborId : graphIter->second) {
            if (closedSet.find(neighborId) != closedSet.end()) {
                std::cout << "Checking neighbor: " << neighborId << " which is already in closed set" << std::endl;
                continue;
            }

            double tentativeGScore = gScore[current.id] + edgeWeight(nodes.at(current.id), nodes.at(neighborId));

            if (gScore.find(neighborId) == gScore.end() || tentativeGScore < gScore[neighborId]) {
                gScore[neighborId] = tentativeGScore;
                cameFrom[neighborId] = current.id;

                AStarNode neighborNode;
                neighborNode.id = neighborId;
                neighborNode.gScore = tentativeGScore;
                neighborNode.fScore = tentativeGScore + heuristic(nodes.at(neighborId), nodes.at(goalId));

                openSet.push(neighborNode);
            }
        }
    }

    // No path found
    return std::vector<int>();
}

int main() {
    std::unordered_map<int, Node> nodes;
    Graph graph;

    parseOSM("map1.osm", nodes, graph);

    for (const auto& entry : graph) {
        int sourceNode = entry.first;

        if (nodes.find(sourceNode) == nodes.end()) {
            std::cerr << "Graph contains node that does not exist: " << sourceNode << '\n';
            return 1;
        }

        for (int neighbor : entry.second) {
            if (nodes.find(neighbor) == nodes.end()) {
                std::cerr << "Graph contains edge to node that does not exist: " << neighbor << '\n';
                return 1;
            }
        }
    }

    int startNodeId = 504338530;
    // long long goalNodeId = 9598507043;
    int goalNodeId = 504348423;

    if (nodes.find(startNodeId) == nodes.end() || nodes.find(goalNodeId) == nodes.end()) {
        std::cerr << "Start or goal node does not exist\n";
        return 1;
    }

    // auto startTime = std::chrono::high_resolution_clock::now();

    try {
        std::vector<int> path = aStarSearch(graph, nodes, startNodeId, goalNodeId);
        if (!path.empty()) {
            std::cout << "Path from " << startNodeId << " to " << goalNodeId << ":\n ";
            for (int nodeId : path) {
                std::cout << nodeId << ' ';
            }
            std::cout << '\n';
        } else {
            std::cout << "No path found.\n";
        }
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range error: " << e.what() << '\n';
        return 1;
    }

    // auto endTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> executionTime = endTime - startTime;

    // std::cout << "Execution time: " << executionTime.count() << " milliseconds" << std::endl;

    return 0;
}
