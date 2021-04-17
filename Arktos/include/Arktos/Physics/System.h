#pragma once

#include "Arktos/Maths/Vec.h"

namespace Arktos::Physics {
    template<class T>
    class System {
    private:
        T timestep;
        Maths::Vec<T> positionX, positionY, positionZ;
        Maths::Vec<T> velocityX, velocityY, velocityZ;
        Maths::Vec<T> accelX, accelY, accelZ;
        Maths::Vec<T> mass;

    public:
        ~System();
        System();
        void pushBody();
    };
}// namespace Arktos::Physics
