#pragma once

#include "Arktos/Maths/BaseVec.h"

// TODO: Document
// TODO: Test

/** @file
 * @brief Class @ref Arktos::Maths::Vec3
 */

namespace Arktos::Maths {
    template<class T> class Vec3 : public BaseVec<T> {
    public:
        constexpr Vec3() noexcept: BaseVec<T>(size_t(3)) {}

        constexpr Vec3(T val) noexcept: BaseVec<T>(val) {}

        constexpr Vec3(T* data) noexcept: BaseVec<T>(data, 3) {}

        constexpr Vec3(T x, T y, T z) noexcept: BaseVec<T>(x, y, z) {}

        template<class U> constexpr explicit Vec3(const BaseVec<T>& other) noexcept: BaseVec<T>(other) {}

        constexpr Vec3(const BaseVec<T>& other) noexcept: BaseVec<T>(other) {}

        // TODO: Test this
        Magnum::Math::Vector3<T> toVector3() {
            return Magnum::Math::Vector3<T>(this->_elements.data());
        }

        T& x() { return BaseVec<T>::_elements[0]; }
        constexpr T x() const { return BaseVec<T>::_elements[0]; } /**< @overload */
        void x(const T x) { BaseVec<T>::_elements[0] = x; }

        T& y() { return BaseVec<T>::_elements[1]; }
        constexpr T y() const { return BaseVec<T>::_elements[1]; } /**< @overload */
        void y(const T y) { BaseVec<T>::_elements[1] = y; }

        T& z() { return BaseVec<T>::_elements[2]; }
        constexpr T z() const { return BaseVec<T>::_elements[2]; } /**< @overload */
        void z(const T z) { BaseVec<T>::_elements[2] = z; }
    };
} // namespace Arktos::Maths