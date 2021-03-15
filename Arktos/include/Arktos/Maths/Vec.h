#pragma once

#include <vector>

namespace Arktos::Maths {
    template<class T>
    class Vec {
    private:
        std::vector<T> elements;

    public:
        Vec() : elements(0){};
        explicit Vec(const uint8_t size) : elements(size){};
        explicit Vec(const std::vector<T> v) : elements(v){};
        Vec(const Vec<T>& v) : elements(v.getData()){};

        ~Vec() = default;

        std::vector<T> getData() const;

        T dot() const;
        T dot(Vec<T> Vec2) const;

        Vec<T> normalized() const;
        void normalise();

    };
}// namespace Arktos::Maths