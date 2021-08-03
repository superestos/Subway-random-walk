#include <chrono>

class Stopwatch
{
private:
    std::chrono::high_resolution_clock::time_point startTime, stopTime;
    size_t totalTime;

public:
    explicit Stopwatch(bool run = false): totalTime(0) {
        if (run) {
            start();
        }
    }

    void start() { 
        startTime = stopTime = std::chrono::high_resolution_clock::now(); 
    }

    void stop() { 
        stopTime = std::chrono::high_resolution_clock::now(); 
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime).count();
    }

    size_t total() const {
        return totalTime;
    }
};