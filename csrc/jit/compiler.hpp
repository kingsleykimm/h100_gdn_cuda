#pragma once
#include <gdn_cuda/utils.h>
#include <nvrtc.h>

#include <fstream>
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/error.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/cache.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/files.hpp>
#include <jit/utils/lazy_init.hpp>
#include <jit/utils/math.hpp>
#include <regex>

class Compiler {
   public:
    inline static fs::path library_root_path;
    inline static fs::path library_include_path;
    inline static fs::path cuda_home;
    inline static std::string library_version;

    static std::string get_library_version() {
        std::vector<char> buffer;
        for (const auto& entry : all_files_in_dir(library_include_path / "gdn_cuda")) {
            std::ifstream stream(entry, std::ios::binary);
            HOST_ASSERT(stream.is_open(), "file not open");
            buffer.insert(buffer.end(), std::istreambuf_iterator<char>(stream),
                          std::istreambuf_iterator<char>());
        }
        return get_hex_digest(buffer);
    }

    static void init_static_vars(fs::path library_root_path_, fs::path cuda_home_) {
        Compiler::library_root_path = library_root_path_;
        Compiler::cuda_home = cuda_home_;
        KernelRuntime::cuda_home = cuda_home_;
        library_include_path = library_root_path_ / "include";
        library_version = get_library_version();
    }

    std::string signature;
    std::string flags;
    fs::path cache_dir_path;

    Compiler() {
        HOST_ASSERT(!library_root_path.empty(), "library_root_path not set");
        HOST_ASSERT(!library_include_path.empty(), "library_include_path not set");
        HOST_ASSERT(!cuda_home.empty(), "cuda_home not set");
        HOST_ASSERT(!library_version.empty(), "library_version not set");

        cache_dir_path = fs::path(get_env<std::string>("HOME")) / ".gdn_cuda";
        if (const auto& manual_cache_dir = get_env<std::string>("JIT_CACHE_DIR");
            !manual_cache_dir.empty()) {
            cache_dir_path = manual_cache_dir;
        }
        signature = "unknown";
        flags =
            fmt::format("-std=c++{} --diag-suppress=177 --ptxas-options=--register-usage-level=10",
                        get_env<int>("JIT_CPP_STD", 17));
    };

    virtual ~Compiler() = default;

    fs::path make_tmp_dir() const { return make_dir(cache_dir_path / "tmp"); }

    fs::path make_tmp_path() const { return make_tmp_dir() / get_uuid(); }

    void write_file(const fs::path& file_name, const std::string& data) const {
        const fs::path tmp_file_path = make_tmp_path();
        std::ofstream out(tmp_file_path, std::ios::binary);
        HOST_ASSERT(out.is_open(), fmt::format("Failed to open temporary file for writing: {}",
                                               tmp_file_path.string())
                                       .c_str());
        out.write(data.data(), data.size());
        HOST_ASSERT(out.good(), "Failed to write data to temporary file");
        out.close();
        std::error_code ec;
        fs::rename(tmp_file_path, file_name, ec);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature =
            fmt::format("{}$${}$${}$${}$${}", name, library_version, signature, flags, code);
        const auto kernel_dir = fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        const fs::path full_path = cache_dir_path / "cache" / kernel_dir;

        if (const auto& runtime_ptr = jit_cache->get_runtime(full_path); runtime_ptr != nullptr) {
            return runtime_ptr;
        }
        make_dir(full_path);
        const auto tmp_path = make_tmp_path();

        compile(code, full_path, tmp_path);

        fs::rename(tmp_path, full_path / "kernel.cubin");

        std::shared_ptr<KernelRuntime> new_runtime = std::make_shared<KernelRuntime>(full_path);
        jit_cache->store_runtime(full_path, new_runtime);

        return new_runtime;
    }

    virtual void compile(const std::string& code, const fs::path kernel_dir_path,
                         const fs::path tmp_cubin_path) const = 0;
};

class NVCC_Compiler : public Compiler {
    fs::path nvcc_path;

   public:
    NVCC_Compiler() {
        nvcc_path = cuda_home / "bin" / "nvcc";
        auto version_cmd = fmt::format("{} --version", nvcc_path.string());
        auto [exitcode, output] = run_command(version_cmd);
        HOST_ASSERT(exitcode == 0, "Error when fetching the nvcc version");
        std::smatch match;
        HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")),
                    "No nvcc version found");
        int major, minor;
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        HOST_WARNING(major == 12 && minor >= 9,
                     "Warning: built with NVCC 12.9, use 12.9 for better performance");
        signature = fmt::format("NVCC-{}-{}", major, minor);
        const auto arch = device_prop->get_major_minor();

        flags = fmt::format(
            "{} -I{} -arch=sm_{} "
            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-abi "
            "-cubin -O3 --expt-related-constexpr --expt-extended-lambda",
            flags, library_include_path.c_str(), device_prop->get_arch(true));
    };

    void compile(const std::string& code, const fs::path kernel_dir_path,
                 const fs::path tmp_cubin_path) const override {
        const fs::path code_path = kernel_dir_path / "kernel.cu";
        write_file(code_path, code);
        std::string command = fmt::format("{} {} -o {} {}", nvcc_path.string(), code_path.string(),
                                          tmp_cubin_path.string(), flags);
        auto [exit_code, output] = run_command(command);
        if (exit_code != 0) {
            printf("NVCC compilation for file %s failed, with output: %s",
                   code_path.string().c_str(), output.c_str());
            HOST_ASSERT(false, "");
        }
        if (get_env("JIT_DEBUG", 0)) {
            printf("%s", output.c_str());
        }
    }
};

class NVRTCCompiler final : public Compiler {
   public:
    NVRTCCompiler() {
        int major, minor;
        NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);

        if (get_env<int>("JIT_DEBUG", 0)) {
            printf("NVRTC version: %d.%d\n", major, minor);
        }
        HOST_ASSERT((major > 12) || (major == 12 && minor >= 3),
                    "NVRTC version must be at least 12.3");

        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        std::string pch_flags;
        if ((major > 12) || (major == 12 && minor >= 8)) {
            pch_flags = "--pch ";
            if (get_env<int>("JIT_DEBUG", 0)) {
                pch_flags += "--pch-verbose=true ";
            }
        }

        flags = fmt::format("{} {} --gpu-architecture=sm_{} -default-device {} --diag-suppress=639",
                            flags, include_dirs,
                            device_prop->get_arch(major >= 13 || (major == 12 && minor >= 9)),
                            pch_flags);
    }

    void compile(const std::string& code, const fs::path kernel_dir_path,
                 const fs::path tmp_cubin_path) const override {
        const fs::path code_path = kernel_dir_path / "kernel.cu";
        write_file(code_path, code);

        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        std::vector<const char*> option_cstrs;
        for (const auto& o : options) {
            option_cstrs.push_back(o.c_str());
        }

        if (get_env<int>("JIT_DEBUG", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt : option_cstrs) {
                printf("%s ", opt);
            }
            printf("\n");
        }

        nvrtcProgram program;
        NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = nvrtcCompileProgram(
            program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        size_t log_size;
        NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env("JIT_DEBUG", 0) || compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS) {
                HOST_ASSERT(log_size > 1, "Log must be meaningful on errors");
            }
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s", compilation_log.c_str());
            }
        }
        size_t cubin_size;
        NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));

        std::string cubin(cubin_size, '\0');
        NVRTC_CHECK(nvrtcGetCUBIN(program, cubin.data()));
        write_file(tmp_cubin_path, cubin);

        NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCC_Compiler>();
    }
});
