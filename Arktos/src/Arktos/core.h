#pragma once

#ifdef AT_PLATFORM_WINDOWS
	#ifdef AT_BUILD_DLL
		#define ARKTOS_API __declspec(dllexport)
	#else
		#define ARKTOS_API __declspec(dllexport)
	#endif
#else
#error Arktos only supports windows! (For now)
#endif