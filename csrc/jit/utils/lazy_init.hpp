#pragma once
#include <functional>
#include <memory>

template <class T>
struct LazyInit {
    std::shared_ptr<T> obj_ptr;
    std::function<std::shared_ptr<T>()> factory_function;

    LazyInit(std::function<std::shared_ptr<T>()> factory_function)
        : factory_function(factory_function) {};

    T* operator->() {
        if (obj_ptr == nullptr) {
            obj_ptr = factory_function();
        }
        return obj_ptr.get();
    }
};
