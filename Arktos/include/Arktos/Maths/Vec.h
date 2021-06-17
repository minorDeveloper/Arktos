#pragma once

/** @file
 * @brief Class @ref Arktos::Maths::Vec
 */

#include "Arktos/Maths/BaseVec.h"

// TODO: Document
// TODO: Test

namespace Arktos::Maths {
    template<class T> class Vec : public BaseVec<T> {
    public:
        constexpr Vec() noexcept: BaseVec<T>() {}

        constexpr explicit Vec(size_t size) noexcept: BaseVec<T>(size) {}

        constexpr Vec(size_t size, T val) noexcept: BaseVec<T>(size, val) {}

        constexpr Vec(T* data, size_t arraySize) noexcept: BaseVec<T>(data, arraySize) {}

        constexpr explicit Vec(std::vector<T>* other) noexcept: BaseVec<T>(other) {}

        template<class ...U> constexpr Vec(T first, U... next) noexcept: BaseVec<T>(first, next...) {}

        template<class U> constexpr explicit Vec(const BaseVec<T>& other) noexcept: BaseVec<T>(other) {}

        constexpr Vec(const BaseVec<T>& other) noexcept: BaseVec<T>(other) {}

        // Appending stuff to the vector
        void appendValue(const T& _value) {
            this->_elements.push_back(_value);
        } // append another vec of the same type (can be different sizes)

        void appendVector(const std::vector<T>& _other) {
            this->_elements.insert(std::end(this->_elements), std::begin(_other), std::end(_other));
        }

        void appendVec(const Vec<T>& _other) {
            this->_elements.insert(std::end(this->_elements), std::begin(_other._elements), std::end(_other._elements));
        }

        void eraseElement(std::size_t _element) {
            assert(_element < this->_elements.size());

            this->_elements.erase(this->_elements.begin() + _element);
        }

        void eraseElements(std::size_t _begin, std::size_t _end) {
            assert(_begin < this->_elements.size() && _end < this->_elements.size() && _begin < _end);

            this->_elements.erase(this->_elements.begin() + _begin, this->_elements.begin() + _end + 1);
        }
    };
} // namespace Arktos::Maths