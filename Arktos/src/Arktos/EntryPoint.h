#pragma once

#ifdef AT_PLATFORM_WINDOWS

extern Arktos::Application* Arktos::CreateApplication();

int main(int argc, char** argv)
{
	Arktos::Log::Init();
	Arktos::Log::GetCoreLogger()->warn("Initilised log!");
	Arktos::Log::GetClientLogger()->info("Initilised log!");

	auto app = Arktos::CreateApplication();
	app->Run();
	delete app;
}

#endif
