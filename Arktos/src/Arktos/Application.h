#pragma once

#include "core.h"

namespace Arktos {

	class ARKTOS_API Application
	{
	public:
		Application();
		virtual~Application();

		void Run();
	};

	// To be defined in CLIENT
	Application* CreateApplication();
}