#pragma once

#include <memory>

#include "Core.h"
#pragma warning(push, 0)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#pragma warning(pop)

// TODO: Document
// TODO: Test

namespace Arktos {

    class Log
    {
    public:
        static void Init();

        static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
        static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }

    private:
        static std::shared_ptr<spdlog::logger> s_CoreLogger;
        static std::shared_ptr<spdlog::logger> s_ClientLogger;
    };
}

// Core logging macros
#if DEBUG
    #define CORE_TRACE(...)     ::Arktos::Log::GetCoreLogger()->trace(__VA_ARGS__)
    #define CORE_INFO(...)      ::Arktos::Log::GetCoreLogger()->info(__VA_ARGS__)
    #define CORE_WARN(...)      ::Arktos::Log::GetCoreLogger()->warn(__VA_ARGS__)
    #define CORE_ERROR(...)      ::Arktos::Log::GetCoreLogger()->error(__VA_ARGS__)
    #define CORE_FATAL(...)      ::Arktos::Log::GetCoreLogger()->fatal(__VA_ARGS__)
#else
    #define CORE_TRACE(...)
    #define CORE_INFO(...)
    #define CORE_WARN(...)
    #define CORE_ERROR(...)
    #define CORE_FATAL(...)
#endif


