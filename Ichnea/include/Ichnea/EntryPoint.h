#pragma once

int main(int argc, char** argv)
{
	Clio::Log::Init();
	Clio::Log::GetCoreLogger()->warn("Initilised log!");
	Clio::Log::GetClientLogger()->info("Initilised log!");

}
