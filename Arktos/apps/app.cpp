

#include <Corrade/Utility/DebugStl.h>

#include <Magnum/Math/Vector.h>

#include "Arktos/Maths/Vec.h"

#include <iostream>
#include <type_traits>
#include <cstdint>

using namespace Arktos::Maths;

int main() {

    Magnum::Math::Vector<3, Magnum::Float> magVec;

    Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
    Vec<Magnum::Float> b = {0.5f, 3.0f, 2.0f, -6.5f};
    Vec<Magnum::Float> c = {1.5f, 1.0f, 5.0f, -10.5f};

    Corrade::Utility::Debug{} << a << "this is a";
    Corrade::Utility::Debug{} << b << "this is b";
    a += b;
    Corrade::Utility::Debug{} << a << "this is a now";
    a = a + b;
    Corrade::Utility::Debug{} << a << "this is a now";

    return 0;
}


//#include "Arktos.h"
//MAGNUM_APPLICATION_MAIN(Arktos::BaseApplication)