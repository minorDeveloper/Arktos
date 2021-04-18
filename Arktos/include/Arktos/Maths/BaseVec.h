#pragma once

/** @file
 * @brief Class @ref Arktos::Maths::BaseVec
 */

#include <vector>
#include <cstddef>
#include <cassert>
#include <type_traits>

#include <Magnum/Math/TypeTraits.h>
#include <Corrade/Utility/Debug.h>

// TODO: Document
// TODO: Test

namespace Arktos::Maths {
    template<class T> class BaseVec {
    public:
        constexpr BaseVec() noexcept: _elements{} {} // Default constructor

        constexpr explicit BaseVec(size_t size) noexcept: _elements(size, 0) {}

        constexpr BaseVec(size_t size, T val) noexcept: _elements(size, val) {} // Constructor with all elements at one value

        constexpr BaseVec(T* data, size_t arraySize) noexcept: _elements{data, data + arraySize} {} // Constructor using an array

        constexpr explicit BaseVec(std::vector<T>* other) noexcept: _elements{other->begin(), other->end()} {}

        template<class ...U> constexpr BaseVec(T first, U... next) noexcept: _elements{first, next...} {}

        //template<class U> constexpr explicit Vec(const Vec<U>& other) noexcept: {} // TODO From a vector of a different type

        T& operator[](std::size_t _pos) {
            assert(_pos < _elements.size());
            return _elements[_pos];
        } // Value at position TODO: Benchmark [] vs .at()

        constexpr T operator[](std::size_t _pos) const {
            assert(_pos < _elements.size());
            return _elements[_pos];
        }

        void set(std::size_t _pos, T _val) {
            assert(_pos < _elements.size());
            _elements[_pos] = _val;
        } // Set value

        // -- Operator overloading

        bool operator==(const BaseVec<T>& other) const {
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

        bool operator!=(const BaseVec<T>& other) const {
            return !operator==(other);
        } // Not equals

        BaseVec<bool> operator< (const BaseVec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            BaseVec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] < other._elements[i]);

            return result;
        } // Less than (need a boolean vector to represent this)

        BaseVec<bool>operator<=(const BaseVec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            BaseVec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] <= other._elements[i]);

            return result;
        } // Less than or equal to

        BaseVec<bool>operator> (const BaseVec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            BaseVec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] < other._elements[i]);

            return result;
        } // Greater than

        BaseVec<bool>operator>=(const BaseVec<T>& other) const {
            assert(_elements.size() == other._elements.size());

            BaseVec<bool> result(std::size_t(_elements.size()));

            for(std::size_t i = 0; i < _elements.size(); ++i)
                result.set(i, _elements[i] <= other._elements[i]);

            return result;
        } // Greater than or equal to

        BaseVec<T>& operator+=(const BaseVec<T>& other) {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] += other._elements[i];

            return *this;
        } // +=

        BaseVec<T> operator+ (const BaseVec<T>& other) const {
            return BaseVec<T>(*this) += other;
        } // +

        BaseVec<T>& operator-=(const BaseVec<T>& other) {
            assert(_elements.size() == other._elements.size());

            for(std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] -= other._elements[i];

            return *this;
        } // -=

        BaseVec<T> operator- (const BaseVec<T>& other) const {
            return BaseVec<T>(*this) -= other;
        } // -

        BaseVec<T>& operator*=(T scalar) {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] *= scalar;

            return *this;
        } // *= (scalar)

        BaseVec<T> operator* (T scalar) const {
            return BaseVec<T>(*this) *= scalar;
        } // * (scalar)

        BaseVec<T>& operator*=(const BaseVec<T>& other) {
            assert(_elements.size() == other.size());

            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] *= other[i];

            return *this;
        } // *= (vector - components wise)

        BaseVec<T> operator* (const BaseVec<T>& other) const {
            return BaseVec<T>(*this) *= other;
        } // * (vector - components wise)

        BaseVec<T>& operator/=(const T scalar) {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] /= scalar;

            return *this;
        } // /= (scalar)

        BaseVec<T> operator/ (const T scalar) {
            return BaseVec<T>(*this) /= scalar;
        } // / (scalar)

        BaseVec<T>& operator/=(const BaseVec<T>& other) {
            assert(_elements.size() == other.size());

            for (std::size_t i = 0; i != _elements.size(); ++i)
                _elements[i] /= other[i];

            return *this;
        } // /= (vector - components wise)

        BaseVec<T> operator/ (const BaseVec<T>& other) {
            return BaseVec<T>(*this) /= other;
        } // / (vector - components wise)

        // -- Checks

        bool isZero() const {
            for (std::size_t i = 0; i != _elements.size(); ++i)
                if (!Magnum::Math::TypeTraits<T>::equalsZero(_elements[i], _elements[i]))
                    return false;

            return true;
        }; // isZero

        bool isNormalised() const {
            return Magnum::Math::TypeTraits<T>::equals(length(), 1.0f);
        }; // isNormalised

        // -- None of these mutate the vector - they return a new vector

        static T dot(const BaseVec<T>& a, const BaseVec<T>& b) {
            assert(a._elements.size() == b._elements.size());
            T result{};

            for(std::size_t i = 0; i < a._elements.size(); ++i)
                result += a._elements[i] * b._elements[i];

            return result;
        }

        T dot(const BaseVec<T>& other) const {
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

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, BaseVec<T>>::type normalised() const {
            return *this*inverseLength();
        } // normalised

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, BaseVec<T>>::type resized(T length) const {
            return *this*(inverseLength())*length;
        }

        BaseVec<T> flipped() const {
            BaseVec<T> temp(_elements.size());

            for(std::size_t i = 1; i <= _elements.size(); ++i)
                temp[i - 1] = _elements[_elements.size() - i];

            return temp;
        } // flipped

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, BaseVec<T>>::type projected(const BaseVec<T>& line) const {
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

        void dottedComponent(const BaseVec<T>& other) {
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

    protected:
        std::vector<T> _elements;
    };


    template<class T> inline BaseVec<T> operator*(typename std::common_type<T>::type scalar, const BaseVec<T>& vector) {
        return vector * scalar;
    }

    template<class T> inline BaseVec<T> operator/(typename std::common_type<T>::type scalar, const BaseVec<T>& vector) {
        BaseVec<T> out(std::size_t(vector.size()));

        for (std::size_t i = 0; i != vector.size(); ++i)
            out[i] = scalar / vector[i];

        return out;
    }

    template<class T> constexpr typename std::enable_if<Magnum::Math::IsScalar<T>::value, T>::type min(T value, T min) {
        return min < value ? min : value;
    }

    template<class T> constexpr typename std::enable_if<Magnum::Math::IsScalar<T>::value, T>::type max(T value, T max) {
        return value < max ? max : value;
    }

    template<class T> inline T BaseVec<T>::min() const {
        T out(_elements[0]);
        for (std::size_t i = 1; i < _elements.size(); ++i)
            out = Maths::min(out, _elements[i]);

        return out;
    }

    template<class T> inline T BaseVec<T>::max() const {
        T out(_elements[0]);
        for (std::size_t i = 1; i < _elements.size(); ++i)
            out = Maths::max(out, _elements[i]);

        return out;
    }

    template<class T> Corrade::Utility::Debug& operator<<(Corrade::Utility::Debug& debug, const BaseVec<T>& value) {
        debug << "Vec(" << Corrade::Utility::Debug::nospace;
        for (std::size_t i = 0; i != value.size(); ++i) {
            if (i != 0) debug << Corrade::Utility::Debug::nospace << ",";
            debug << value[i];
        }
        return debug << Corrade::Utility::Debug::nospace << ")";
    }// debug output
} // namespace Arktos::Maths