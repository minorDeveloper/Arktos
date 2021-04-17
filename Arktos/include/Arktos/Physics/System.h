#pragma once

#include "Arktos/Maths/Vec.h"
#include "Arktos/Maths/Vec3.h"

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
        System(size_t bodies);

        void pushBody(const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity, const T bodyMass);
        void popBody(size_t bodyID);

        Maths::Vec3<T> centreOfMass();
        Maths::Vec3<T> centreOfVelocity();

        T potentialEnergy();
        T kineticEnergy();
        T totalEnergy() { return potentialEnergy() + kineticEnergy(); }
    };
}// namespace Arktos::Physics
