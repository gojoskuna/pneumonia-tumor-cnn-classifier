#include <iostream>
#include <algorithm>
using namespace std;

class LamportClock {
private:
    int pid;
    int clock;

public:
    LamportClock(int id) : pid(id), clock(0) {}

    void internalEvent() {
        clock++;
        cout << "P" << pid << " internal event | clock = " << clock << endl;
    }

    int sendEvent() {
        clock++;
        cout << "P" << pid << " sends message | clock = " << clock << endl;
        return clock;
    }

    void receiveEvent(int timestamp) {
        clock = max(clock, timestamp) + 1;
        cout << "P" << pid << " receives message | clock = " << clock << endl;
    }
};

int main() {
    LamportClock p1(1), p2(2);

    int t = p1.sendEvent();
    p2.receiveEvent(t);

    return 0;
}
