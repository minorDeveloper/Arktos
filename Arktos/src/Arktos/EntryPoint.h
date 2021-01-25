#pragma once

#ifdef AT_PLATFORM_WINDOWS

extern Arktos::Application* Arktos::CreateApplication();

int main(int argc, char** argv)
{
	printf("Arktos Engine");
	auto app = Arktos::CreateApplication();
	app->Run();
	delete app;
}

#endif
