#pragma once

#include <memory>

#include "core.h"
#pragma warning(push, 0)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#pragma warning(pop)

namespace Ichnea {

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