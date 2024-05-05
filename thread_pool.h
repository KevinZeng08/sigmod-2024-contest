#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>

class ThreadPool {
 public:
    ThreadPool(size_t);
    template <class F, class... Args>
    auto
    push(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

    static std::shared_ptr<ThreadPool> GetBuildPool() {
        std::lock_guard<std::mutex> lock(build_pool_mutex_);
        if (build_pool_ == nullptr) {
            build_pool_ = std::make_shared<ThreadPool>(std::thread::hardware_concurrency());
        }
        return build_pool_;
    }

    static std::shared_ptr<ThreadPool> GetSearchPool() {
        std::lock_guard<std::mutex> lock(search_pool_mutex_);
        if (search_pool_ == nullptr) {
            search_pool_ = std::make_shared<ThreadPool>(std::thread::hardware_concurrency());
        }
        return search_pool_;
    }

 private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    inline static std::mutex build_pool_mutex_;
    inline static std::shared_ptr<ThreadPool> build_pool_ = nullptr;

    inline static std::mutex search_pool_mutex_;
    inline static std::shared_ptr<ThreadPool> search_pool_ = nullptr;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
}

// add new work item to the pool
template <class F, class... Args>
auto
ThreadPool::push(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) worker.join();
}

