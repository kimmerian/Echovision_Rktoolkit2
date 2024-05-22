//
// Created by root on 3/12/24.
//


#include <mutex>
#include <condition_variable>
#include <queue>
#include <exception>
#include <atomic>

template<typename T>
class CQueue {
private:
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> terminate{ false };

public:
    void push(const T& value) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (terminate) return;
            queue.push(value);
        }
        cv.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !queue.empty() || terminate; });
        if (queue.empty()) return false;
        value = std::move(queue.front());
        queue.pop();
        return true;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            terminate = true;
        }
        cv.notify_all();
    }

    bool shouldTerminate() const {
        return terminate;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};
