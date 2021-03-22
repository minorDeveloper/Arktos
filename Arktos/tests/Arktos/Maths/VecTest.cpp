
#include <Corrade/TestSuite/Tester.h>

#include <Arktos/Maths/Vec.h>
#include <Arktos/Log.h>

namespace Arktos::Maths::Test { namespace {
    struct VecTest: Corrade::TestSuite::Tester {
        explicit VecTest();

        void addition();
        void scalarMult();
        void dot();
    };

    VecTest::VecTest() {
        addTests({&VecTest::addition,
                  &VecTest::scalarMult,
                  &VecTest::dot});
    }

    void VecTest::addition() {
        Arktos::Maths::Vec<double> vector;
        CORRADE_COMPARE(2,2);
    }

    void VecTest::dot() {
        CORRADE_EXPECT_FAIL("Dot product has not yet been implemented");
        CORRADE_COMPARE(2,3);
    }

    void VecTest::scalarMult() {
        CORRADE_COMPARE(2,3);
    }
}}

CORRADE_TEST_MAIN(Arktos::Maths::Test::VecTest)