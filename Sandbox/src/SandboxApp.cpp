#include "Arktos.h"

class Sandbox : public Arktos::Application
{
public:
	Sandbox()
	{

	}

	~Sandbox()
	{

	}
};

Arktos::Application* Arktos::CreateApplication()
{
	return new Sandbox();
}