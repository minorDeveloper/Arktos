#pragma once

#include "Arktos/Maths/BaseVec.h"

/** @file
 * @brief Class @ref Arktos::Maths::Vec3
 */

namespace Arktos::Maths {
    template<class T> class Vec3 : public BaseVec {
    public:
        constexpr Vec3() noexcept: BaseVec<T>(size_t(3)) {}

        constexpr Vec3(T val) noexcept: BaseVec<T>(val) {}

        constexpr Vec3(T* data) noexcept: BaseVec<T>(data, 3) {}

        constexpr Vec3(T x, T y, T z) noexcept: BaseVec<T>(x, y, z) {}

        template<class U> constexpr explicit Vec3(const BaseVec<T>& other) noexcept: BaseVec<T>(other) {}

        constexpr Vec3(const Vec3<T>& other) noexcept: Vec3<T>(other) {}
    };
} // namespace Arktos::Maths