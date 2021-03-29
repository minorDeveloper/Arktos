#pragma once

/** @file
 * @brief Class @ref Arktos::Maths::Vec
 */

#include <vector>
#include <cstddef>
#include <cassert>
#include <type_traits>

#include <Magnum/Math/TypeTraits.h>
#include <Corrade/Utility/Debug.h>


namespace Arktos::Maths {

    template<class T> class Vec {
    public:
        constexpr Vec() noexcept: _elements{} {} // Default constructor, size of size

        constexpr Vec(size_t size) noexcept: _elements(size, 0) {}

        constexpr Vec(size_t size, T val) noexcept: _elements(size, val) {}// Constructor with all elements at one value

        constexpr Vec(T* data, size_t arraySize) noexcept: _elements{data, data + arraySize} {}// Constructor using an array

        constexpr Vec(std::vector<T>* other) noexcept: _elements{other->begin(), other->end()} {}// TODO see if this is actually better for large vectors

        template<class ...U> constexpr Vec(T first, U... next) noexcept: _elements{first, next...} {} // TODO Component wise

        //template<class U> constexpr explicit Vec(const Vec<U>& other) noexcept: {} // TODO From a vector of a different type

        constexpr Vec(const Vec<T>&) noexcept = default;// Copy constructor

        T& operator[](std::size_t _pos) {
            assert(_pos < _elements.size());
            return _elements[_pos];
        }// Value at position TODO: Benchmark [] vs .at()

        constexpr T operator[](std::size_t _pos) const {
            assert(_pos < _elements.size());
            return _elements[_pos];
        }

        void set(std::size_t _pos, T _val) {
            assert(_pos < _elements.size());
            _elements[_pos] = _val;
        }// Set value


        // -- Operator overloading

        bool operator==(const Vec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i < _elements.size(); ++i)
                if(!Magnum::Math::TypeTraits<T>::equals(_elements[i], other._elements[i])) return false;

            return true;
        } // Equality

        template<class U = T> typename std::enable_if<std::is_same<bool, U>::value, bool>::type operator==(const bool& other) const {

            for (std::size_t i = 0; i < _elements.size(); ++i)
                if(!Magnum::Math::TypeTraits<T>::equals(_elements[i], other)) return false;

            return true;
        }


        bool operator!=(const Vec<T>& other) const {
            return !operator==(other);
        } // Not equals

        Vec<bool> operator< (const Vec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            Vec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] < other._elements[i]);

            return result;
        } // Less than (need a boolean vector to represent this)

        Vec<bool>operator<=(const Vec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            Vec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] <= other._elements[i]);

            return result;
        } // Less than or equal to

        Vec<bool>operator> (const Vec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            Vec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] < other._elements[i]);

            return result;
        } // Greater than

        Vec<bool>operator>=(const Vec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            Vec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] <= other._elements[i]);

            return result;
        } // Greater than or equal to

        Vec<T>& operator+=(const Vec<T>& other) {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] += other._elements[i];

            return *this;
        } // +=

        Vec<T> operator+ (const Vec<T>& other) const {
            return Vec(*this) += other;
        } // +

        Vec<T>& operator-=(const Vec<T>& other) {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] -= other._elements[i];

            return *this;
        } // -=

        Vec<T> operator- (const Vec<T>& other) const {
            return Vec(*this) -= other;
        } // -

        Vec<T>& operator*=(T scalar) {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] *= scalar;

            return *this;
        } // *= (scalar)

        Vec<T> operator* (T scalar) const {
            return Vec(*this) *= scalar;
        } // * (scalar)

        Vec<T>& operator*=(const Vec<T>& other) {
            assert(_elements.size() == other.size());

            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] *= other[i];

            return *this;
        } // *= (vector - components wise)

        Vec<T> operator* (const Vec<T>& other) const {
            return Vec(*this) *= other;
        } // * (vector - components wise)

        Vec<T>& operator/=(const T scalar) {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] /= scalar;

            return *this;
        } // /= (scalar)

        Vec<T> operator/ (const T scalar) {
            return Vec(*this) /= scalar;
        } // / (scalar)

        Vec<T>& operator/=(const Vec<T>& other) {
            assert(_elements.size() == other.size());

            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] /= other[i];

            return *this;
        } // /= (vector - components wise)

        Vec<T> operator/ (const Vec<T>& other) {
            return Vec(*this) /= other;
        } // / (vector - components wise)

        // -- Checks

        bool isZero() const {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                if (!Magnum::Math::TypeTraits<T>::equalsZero(_elements[i], _elements[i]))
                    return false;

            return true;
        }; // isZero TODO

        bool isNormalised() const {
            return Magnum::Math::TypeTraits<T>::equals(length(), 1.0f);
        }; // isNormalised

        // -- None of these mutate the vector - they return a new vector

        static T dot(const Vec<T>& a, const Vec<T>& b) {
            assert(a._elements.size() == b._elements.size());
            T result{};

            for(std::size_t i = 0; i < a._elements.size(); ++i)
                result += a._elements[i] * b._elements[i];

            return result;
        }

        T dot(const Vec<T>& other) const {
            return dot(*this, other);
        } // dot product (return a new Vec)

        T dot() const {
            return dot(*this, *this);
        }

        T length() const {
            return T(std::sqrt(dot()));
        }// length

        T inverseLength() const {
            return T(1)/length();
        } // inverse length

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vec<T>>::type normalised() const {
            return *this*inverseLength();
        } // normalised

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vec<T>>::type resized(T length) const {
            return *this*(inverseLength())*length;
        }

        Vec<T> flipped() const {
            Vec<T> temp(_elements.size());

            for(std::size_t i = 1; i <= _elements.size(); ++i)
                temp[i - 1] = _elements[_elements.size() - i];

            return temp;
        } // flipped

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vec<T>>::type projected(const Vec<T>& line) const {
            return line*dot(line)/line.dot();
        }

        T summed() const {
            T out{};

            for (std::size_t i = 0; i < _elements.size(); ++i)
                out += _elements[i];

            return out;
        } // summed

        T min() const; // min

        T max() const; // max

        std::size_t size() const {
            return _elements.size();
        }

        // -- The following mutate the vector itself and therefore return void

        void dottedComponent(const Vec<T>& other) {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i < _elements.size(); ++i)
                _elements[i] *= other._elements[i];
        } // dotted

        void normalise() {
            _elements = this->normalised()._elements;
            //_elements = normalised();
        } // normalise

        void flip() {
            _elements = this->flipped()._elements;
        } // flip

        // Debug stuff

        std::vector<T> toVector() { return _elements; } // toVector

        // Appending stuff from to the vector

        void appendValue(const T& _value) {
            _elements.push_back(_value);
        } // append another vec of the same type (can be different sizes)

        void appendVector(const std::vector<T>& _other) {
            _elements.insert(std::end(_elements), std::begin(_other), std::end(_other));
        }

        void appendVec(const Vec<T>& _other) {
            _elements.insert(std::end(_elements), std::begin(_other._elements), std::end(_other._elements));
        }

        void eraseElement(std::size_t _element) {
            assert(_element < _elements.size());

            _elements.erase(_elements.begin() + _element);
        }

        void eraseElements(std::size_t _begin, std::size_t _end) {
            assert(_begin < _elements.size() && _end < _elements.size() && _begin < _end);

            _elements.erase(_elements.begin() + _begin, _elements.begin() + _end + 1);
        }
    private:
        std::vector<T> _elements;
    };


    template<class T> inline Vec<T> operator*(typename std::common_type<T>::type scalar, const Vec<T>& vector) {
        return vector * scalar;
    }

    template<class T> inline Vec<T> operator/(typename std::common_type<T>::type scalar, const Vec<T>& vector) {
        Vec<T> out(std::size_t(vector.size()));

        for (std::size_t i = 0; i != vector.size(); ++i)
            out[i] = scalar/vector[i];

        return out;
    }

    template<class T> constexpr typename std::enable_if<Magnum::Math::IsScalar<T>::value, T>::type min(T value, T min) {
        return min < value ? min : value;
    }

    template<class T> constexpr typename std::enable_if<Magnum::Math::IsScalar<T>::value, T>::type max(T value, T max) {
        return value < max ? max : value;
    }

    template<class T> inline T Vec<T>::min() const {
        T out(_elements[0]);
        for (std::size_t i = 1; i < _elements.size(); ++i)
            out = Maths::min(out, _elements[i]);

        return out;
    }

    template<class T> inline T Vec<T>::max() const {
        T out(_elements[0]);
        for (std::size_t i = 1; i < _elements.size(); ++i)
            out = Maths::max(out, _elements[i]);

        return out;
    }

    template<class T> Corrade::Utility::Debug& operator<<(Corrade::Utility::Debug& debug, const Vec<T>& value) {
        debug << "Vec(" << Corrade::Utility::Debug::nospace;
        for (std::size_t i = 0; i != value.size(); ++i) {
            if (i != 0) debug << Corrade::Utility::Debug::nospace << ",";
            debug << value[i];
        }
        return debug << Corrade::Utility::Debug::nospace << ")";
    }// debug output
}// namespace Arktos::Maths