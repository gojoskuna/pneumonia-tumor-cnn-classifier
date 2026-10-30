#include <bits/stdc++.h>
using namespace std;

class Message {
public:
    int sender_id;
    string text;
    vector<int> vector_clock;

    Message(int sender, string txt, vector<int> vc)
        : sender_id(sender), text(txt), vector_clock(vc) {}
};

class Node {
public:
    int node_id;
    vector<int> vector_clock;

    Node(int id, int total_nodes) {
        node_id = id;
        vector_clock.resize(total_nodes, 0);
    }

    void send_message(Node &receiver, string message_text) {
        // Increment sender's clock
        vector_clock[node_id]++;

        // Create message with a copy of vector clock
        Message msg(node_id, message_text, vector_clock);

        // Send message
        receiver.receive_message(msg);
    }

    void receive_message(const Message &message) {
        // Update vector clock using max rule
        for (int i = 0; i < vector_clock.size(); i++) {
            vector_clock[i] = max(vector_clock[i], message.vector_clock[i]);
        }

        // Increment receiver's own clock
        vector_clock[node_id]++;

        cout << "Node " << node_id << " received message: "
             << message.text << endl;
        cout << "Updated vector clock: ";
        print_clock();
    }

    void print_clock() const {
        cout << "[ ";
        for (int v : vector_clock)
            cout << v << " ";
        cout << "]\n";
    }

    void display() const {
        cout << "Node " << node_id << " - Vector Clock: ";
        print_clock();
    }
};

class DistributedSystem {
public:
    vector<Node> nodes;

    DistributedSystem(int num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            nodes.emplace_back(i, num_nodes);
        }
    }

    void simulate() {
        // Node 0 → Node 1
        nodes[0].send_message(nodes[1], "Hello from Node 0");

        // Node 1 → Node 2
        nodes[1].send_message(nodes[2], "Hello from Node 1");

        // Node 2 → Node 0
        nodes[2].send_message(nodes[0], "Hello from Node 2");

        // Final clocks
        cout << "\nFinal Vector Clocks:\n";
        for (const auto &node : nodes)
            node.display();
    }
};

int main() {
    DistributedSystem system(3);
    system.simulate();
    return 0;
}

