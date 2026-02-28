#pragma once
#include <cuda_runtime.h>

#include <jit/utils/lazy_init.hpp>
#include <memory>

#include "error.hpp"

struct DeviceProps {
   public:
    std::shared_ptr<cudaDeviceProp> prop_ptr;
    int device;

    DeviceProps() {};

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (prop_ptr == nullptr) {
            int device;
            cudaDeviceProp new_prop;
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&new_prop, device));
            prop_ptr = std::make_shared<cudaDeviceProp>(new_prop);
        }
        return prop_ptr;
    }

    std::pair<int, int> get_major_minor() {
        const auto prop = get_prop();
        return std::make_pair(prop->major, prop->minor);
    }

    std::string get_arch(const bool& include_letter = false) {
        auto [major, minor] = get_major_minor();
        return std::to_string(major * 10 + minor) + (include_letter ? "a" : "");
    }

    int get_num_sms() { return get_prop()->multiProcessorCount; }

    size_t get_smem_size() { return get_prop()->sharedMemPerBlock; }

    std::pair<size_t, size_t> get_max_clusters() {
        auto [major, minor] = get_major_minor();
        if (major >= 9) {
            if (major == 9) {
                return std::make_pair(8, 16);
            }
        } else {
            HOST_ERROR("get_max_clusters: Cluster launch is not supported on this device");
        }
        return std::make_pair(0, 0);
    }
};

static auto device_prop = LazyInit<DeviceProps>(
    []() -> std::shared_ptr<DeviceProps> { return std::make_shared<DeviceProps>(); });
