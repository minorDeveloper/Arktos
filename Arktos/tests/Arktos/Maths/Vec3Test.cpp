
#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/DebugStl.h>
#include <Magnum/Math/TypeTraits.h>

#include "Arktos/Maths/Vec3.h"

using namespace Arktos::Maths;

namespace Arktos::Maths::Test { namespace {
    struct Vec3Test: Corrade::TestSuite::Tester {
        explicit Vec3Test();

        void yAccess();

    };

    Test::Vec3Test::Vec3Test() {
        addTests({
                &Vec3Test::yAccess,
        });

    }

    void Test::Vec3Test::yAccess() {
        Vec3<Magnum::Float> a = {1.0f, 2.0f, 3.0f};

        CORRADE_COMPARE(a.y(), 2.0f);
    }
}}
CORRADE_TEST_MAIN(Arktos::Maths::Test::Vec3Test)