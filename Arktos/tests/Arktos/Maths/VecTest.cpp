
#include <sstream>
#include <vector>

#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/DebugStl.h>
#include <Magnum/Math/TypeTraits.h>

#include <Arktos/Maths/Vec.h>

using namespace Arktos::Maths;

namespace Arktos::Maths::Test { namespace {
    struct VecTest: Corrade::TestSuite::Tester {
        explicit VecTest();

        void construct();
        void constructDefault();
        void constructSize();
        void constructFromValue();
        void constructFromArray();
        void constructFromVector();
        void constructCopy();

        void arrayAccess();
        void setValue();

        void equality();
        void notEquality();

        void comparison();

        void addSubtract();

        void scalarMultiply();
        void vectorMultiply();

        void scalarDivide();
        void vectorDivide();

        void isZero();
        void isNormalised();

        void dot();
        void length();
        void inverseLength();
        void normalised();
        void flipped();
        void projected();
        void summed();
        void min();
        void max();
        void size();

        void normalise();
        void flip();

        void toVector();

        void appendValue();
        void appendVector();
        void appendVec();

        void eraseElement();
        void eraseElements();

        void debug();

    };

    Test::VecTest::VecTest() {
        addTests({&VecTest::construct,
                  &VecTest::constructDefault,
                  &VecTest::constructSize,
                  &VecTest::constructFromArray,
                  &VecTest::constructFromVector,
                  &VecTest::constructFromValue,
                  &VecTest::constructCopy,
                  &VecTest::arrayAccess,
                  &VecTest::setValue,
                  &VecTest::equality,
                  &VecTest::notEquality,
                  &VecTest::comparison,
                  &VecTest::addSubtract,
                  &VecTest::scalarMultiply,
                  &VecTest::vectorMultiply,
                  &VecTest::scalarDivide,
                  &VecTest::vectorDivide,
                  &VecTest::isZero,
                  &VecTest::isNormalised,
                  &VecTest::dot,
                  &VecTest::length,
                  &VecTest::inverseLength,
                  &VecTest::normalised,
                  &VecTest::flipped,
                  &VecTest::projected,
                  &VecTest::summed,
                  &VecTest::min,
                  &VecTest::max,
                  &VecTest::size,
                  &VecTest::normalise,
                  &VecTest::flip,
                  &VecTest::toVector,
                  &VecTest::appendValue,
                  &VecTest::appendVector,
                  &VecTest::appendVec,
                  &VecTest::eraseElement,
                  &VecTest::eraseElements,
                  &VecTest::debug});
    }
    
    void Test::VecTest::construct() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};

        CORRADE_COMPARE(a, Vec(1.0f, -2.0f, 3.0f, -4.0f));
    }

    void Test::VecTest::constructDefault() {
        Vec<Magnum::Float> a;

        CORRADE_COMPARE(a, a);
    }

    void Test::VecTest::constructSize() {
        Vec<Magnum::Float> a(size_t(3));

        CORRADE_COMPARE(a, Vec(0.0f, 0.0f, 0.0f));
    }

    void Test::VecTest::constructFromValue() {
        Vec<Magnum::Float> a(size_t(4),7.0f);

        CORRADE_COMPARE(a, Vec(7.0f, 7.0f, 7.0f, 7.0f));
    }

    void Test::VecTest::constructFromArray() {
        Magnum::Int rawData[] = {1, 2, 3};
        Vec<Magnum::Int> a(rawData, 3);

        CORRADE_COMPARE(a, Vec(1, 2, 3));
    }

    void Test::VecTest::constructFromVector() {
        std::vector<Magnum::Int> rawVector = {4, 5, 6};
        Vec<Magnum::Int> a(&rawVector);

        CORRADE_COMPARE(a, Vec(4, 5, 6));
    }

    void Test::VecTest::constructCopy() {
        Vec<Magnum::Int> a(1, 3, 5, 7);
        Vec<Magnum::Int> b(a);

        CORRADE_COMPARE(b, Vec(1, 3, 5, 7));
    }

    void Test::VecTest::arrayAccess() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};

        CORRADE_COMPARE(a[0], 1.0f);
        CORRADE_COMPARE(a[3], -4.0f);
    }

    void Test::VecTest::setValue() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
        a[1] = 7.0f;

        CORRADE_COMPARE(a[1], 7.0f);
        CORRADE_COMPARE(a, Vec(1.0f, 7.0f, 3.0f, -4.0f));
    }

    void Test::VecTest::equality() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
        Vec<Magnum::Float> b = {1.0f, -2.0f, 7.0f, -4.0f};

        CORRADE_VERIFY(a == a);

        CORRADE_VERIFY(Vec(1.0f, -2.0f, 3.0f, -4.0f) == Vec(1.0f + Magnum::Math::TypeTraits<Magnum::Float>::epsilon()/2, -2.0f, 3.0f, -4.0f));
    }

    void Test::VecTest::notEquality() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
        Vec<Magnum::Float> b = {1.0f, -2.0f, 7.0f, -4.0f};

        CORRADE_VERIFY(a != b);

        CORRADE_VERIFY(Vec(1.0f, -2.0f, 3.0f, -4.0f) != Vec(1.0f + Magnum::Math::TypeTraits<Magnum::Float>::epsilon()*2, -2.0f, 3.0f, -4.0f));
    }

    void Test::VecTest::comparison() {

        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
        Vec<Magnum::Float> b = {0.0f, -3.0f, 2.0f, -6.0f};
        Vec<Magnum::Float> c = {0.0f, -3.0f, 2.0f, -3.0f};

        CORRADE_VERIFY(((b <  a)== true));
        CORRADE_VERIFY(b <= a == true);
        CORRADE_VERIFY(a <  a == false);
        CORRADE_VERIFY(b >= b == true);
        CORRADE_VERIFY(b >= c == true);
        CORRADE_VERIFY((b >=  c) == true);
    }

    void Test::VecTest::addSubtract() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.0f};
        Vec<Magnum::Float> b = {0.5f, 3.0f, 2.0f, -6.5f};
        Vec<Magnum::Float> c = {1.5f, 1.0f, 5.0f, -10.5f};

        CORRADE_COMPARE(a + b, c);
        CORRADE_COMPARE(c - b, a);
    }


    void Test::VecTest::scalarMultiply() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.5f};
        Vec<Magnum::Float> b = {-0.5f, 1.0f, -1.5f, 2.25f};

        CORRADE_COMPARE(-0.5f * a, b);
        CORRADE_COMPARE(a * -0.5f, b);
    }

    void Test::VecTest::vectorMultiply() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.5f};
        Vec<Magnum::Float> b = {2.0f, 1.5f, 3.0f, -2.0f};
        Vec<Magnum::Float> c = {2.0f, -3.0f, 9.0f, 9.0f};

        CORRADE_COMPARE(a * b, c);
        CORRADE_COMPARE(b * a, c);
    }

    void Test::VecTest::scalarDivide() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.5f};
        Vec<Magnum::Float> b = {-0.5f, 1.0f, -1.5f, 2.25f};

        CORRADE_COMPARE(b / -0.5f, a);
    }

    void Test::VecTest::vectorDivide() {
        Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.5f};
        Vec<Magnum::Float> b = {2.0f, 1.5f, 3.0f, -2.0f};
        Vec<Magnum::Float> c = {2.0f, -3.0f, 9.0f, 9.0f};

        CORRADE_COMPARE(c / b, a);
    }

    void Test::VecTest::isZero() {
        const Vec<Magnum::Float> a = {3.0f, 4.0f, 5.0f};
        const Vec<Magnum::Float> b = {0.0f, 0.0f, 0.0f};
        CORRADE_VERIFY(!a.isZero());
        CORRADE_VERIFY(b.isZero());
    }

    void Test::VecTest::isNormalised() {
        const Vec<Magnum::Float> a = {3.0f, 4.0f, 5.0f};
        const Vec<Magnum::Float> aNorm = {0.42426406871192851f, 0.56568542494923801f, 0.70710678118654746f};
        CORRADE_VERIFY(!a.isNormalised());
        CORRADE_VERIFY(aNorm.isNormalised());
    }

    void Test::VecTest::dot() {
        const Vec<Magnum::Float> a = {1.0f, -2.0f, 3.0f, -4.5f};
        const Vec<Magnum::Float> b = {2.0f, 1.5f, 3.0f, -2.0f};

        CORRADE_COMPARE(a.dot(), 34.25f);
        CORRADE_COMPARE(a.dot(b), 17.0f);
        CORRADE_COMPARE(Vec<Magnum::Float>::dot(a,b), 17.0f);
    }

    void Test::VecTest::length() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, 4.0f).length(), 5.0f);
    }

    void Test::VecTest::inverseLength() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, 4.0f).inverseLength(), 0.2f);
    }

    void Test::VecTest::normalised() {
        const auto vec = Vec<Magnum::Float>(3.0f, 4.0f, 5.0f).normalised();
        CORRADE_COMPARE(vec, Vec<Magnum::Float>(0.42426406871192851f, 0.56568542494923801f, 0.70710678118654746f));
        CORRADE_COMPARE(vec.length(), 1.0f);
    }

    void Test::VecTest::flipped() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, 4.0f, 5.0f).flipped(), Vec<Magnum::Float>(5.0f, 4.0f, 3.0f));
    }

    void Test::VecTest::projected() {
        Vec<Magnum::Float> line(1.0f, -1.0f, 0.5f);
        Vec<Magnum::Float> projected = Vec<Magnum::Float>(1.0f, 2.0f, 3.0f).projected(line);
        CORRADE_COMPARE(projected, Vec<Magnum::Float>(0.222222f, -0.222222f, 0.111111f));
        CORRADE_COMPARE(projected.normalised(), projected.normalised());
    }

    void Test::VecTest::summed() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, 4.0f, 5.0f).summed(), 12.0f);
    }

    void Test::VecTest::min() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, 4.0f, 5.0f).min(), 3.0f);
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, -4.0f, 5.0f).min(), -4.0f);
    }

    void Test::VecTest::max() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, -4.0f, 5.0f).max(), 5.0f);
    }

    void Test::VecTest::size() {
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, -4.0f, 5.0f).size(), std::size_t(3));
    }

    void Test::VecTest::normalise() {
        Vec<Magnum::Float> a = {3.0f, 4.0f, 5.0f};
        a.normalise();
        CORRADE_COMPARE(a, Vec<Magnum::Float>(0.42426406871192851f, 0.56568542494923801f, 0.70710678118654746f));
        CORRADE_COMPARE(a.length(), 1.0f);
    }

    void Test::VecTest::flip() {
        Vec<Magnum::Float> a = {3.0f, 4.0f, 5.0f};
        a.flip();
        CORRADE_COMPARE(a, Vec<Magnum::Float>(5.0f, 4.0f, 3.0f));
    }

    void Test::VecTest::toVector() {
        std::vector<Magnum::Float> vec = {3.0f, -4.0f, 5.0f};
        CORRADE_COMPARE(Vec<Magnum::Float>(3.0f, -4.0f, 5.0f).toVector(), vec);
    }

    void Test::VecTest::appendValue() {
        Vec<Magnum::Float> a = {3.0f, -4.0f, 5.0f};
        a.appendValue(7.0f);
        CORRADE_COMPARE(a, Vec<Magnum::Float>(3.0f, -4.0f, 5.0f, 7.0f));
    }

    void Test::VecTest::appendVector() {
        Vec<Magnum::Float> a = {3.0f, -4.0f, 5.0f};
        std::vector<Magnum::Float> b = {6.0f, 7.0f, -8.0f};
        a.appendVector(b);
        CORRADE_COMPARE(a, Vec<Magnum::Float>(3.0f, -4.0f, 5.0f, 6.0f, 7.0f, -8.0f));
    }

    void Test::VecTest::appendVec() {
        Vec<Magnum::Float> a = {3.0f, -4.0f, 5.0f};
        Vec<Magnum::Float> b = {6.0f, 7.0f, -8.0f};
        a.appendVec(b);
        CORRADE_COMPARE(a, Vec<Magnum::Float>(3.0f, -4.0f, 5.0f, 6.0f, 7.0f, -8.0f));
    }

    void Test::VecTest::eraseElement() {
        Vec<Magnum::Float> a = {3.0f, -4.0f, 5.0f};
        a.eraseElement(1);
        CORRADE_COMPARE(a, Vec<Magnum::Float>(3.0f, 5.0f));
    }

    void Test::VecTest::eraseElements() {
        Vec<Magnum::Float> a = {3.0f, -4.0f, 7.0f, 8.0f, 5.0f};
        a.eraseElements(1, 3);
        CORRADE_COMPARE(a, Vec<Magnum::Float>(3.0f, 5.0f));
    }

    void Test::VecTest::debug() {
        std::ostringstream o;
        Corrade::Utility::Debug{&o} << Vec<Magnum::Float>(4.0f, 0.5f, 6.0f, -15.0f);
        CORRADE_COMPARE(o.str(), "Vec(4, 0.5, 6, -15)\n");

        o.str({});
        Vec<Magnum::Int> a(std::size_t(4));
        Corrade::Utility::Debug{&o}  << "a" << a;
        CORRADE_COMPARE(o.str(), "a Vec(0, 0, 0, 0)\n");
    }
}}

CORRADE_TEST_MAIN(Arktos::Maths::Test::VecTest)