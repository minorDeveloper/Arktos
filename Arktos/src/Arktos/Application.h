#pragma once

#include "core.h"

namespace Arktos {

	class Application
	{
	public:
		Application();
		virtual~Application();

		void Run();
	};

	// To be defined in CLIENT
	Application* CreateApplication();
}