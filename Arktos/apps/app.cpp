
#include <iostream>

#include "Arktos/Log.h"

int main() {
    Arktos::Log::Init();
    std::cout << "Hello everybody!" << std::endl;
    return 0;
}