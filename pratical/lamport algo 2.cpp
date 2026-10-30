#include <iostream>
#include <queue>
#include <vector>
using namespace std;

struct Request {
    int timestamp;
    int pid;

    bool operator>(const Request& other) const {
        if (timestamp == other.timestamp)
            return pid > other.pid;
        return timestamp > other.timestamp;
    }
};

class Process {
private:
    int pid;
    int clock;
    int totalProcesses;
    int replies;
    priority_queue<Request, vector<Request>, greater<Request>> requestQueue;

public:
    Process(int id, int total)
        : pid(id), clock(0), totalProcesses(total), replies(0) {}

    Request requestCS() {
        clock++;
        Request req{clock, pid};
        requestQueue.push(req);
        cout << "P" << pid << " REQUEST CS at time " << clock << endl;
        return req;
    }

    void receiveRequest(Request req) {
        clock = max(clock, req.timestamp) + 1;
        requestQueue.push(req);
        cout << "P" << pid << " received REQUEST from P" << req.pid << endl;
    }

    void receiveReply() {
        replies++;
    }

    bool canEnterCS() {
        return replies == totalProcesses - 1 &&
               requestQueue.top().pid == pid;
    }

    void enterCS() {
        cout << "P" << pid << " ENTERS critical section" << endl;
    }

    void releaseCS() {
        requestQueue.pop();
        replies = 0;
        cout << "P" << pid << " RELEASES critical section" << endl;
    }
};
