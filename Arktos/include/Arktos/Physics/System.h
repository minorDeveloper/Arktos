#pragma once

#include "Arktos/Maths/Vec.h"
#include "Arktos/Maths/Vec3.h"

namespace Arktos::Physics {
    enum class TimeStep { FIXED, VARIABLE};
    enum class Integrator { DIRECT, RK4 };

    template<class T>
    class System {
    public:
        ~System();
        System();
        System(const size_t bodies, const TimeStep stepType);

        // Frame shifting
        const Maths::Vec3<T> centreOfMass();
        const Maths::Vec3<T> centreOfVelocity();
        bool toCoMoving();

        // Energy conservation
        const T potentialEnergy();
        const T kineticEnergy();
        const T totalEnergy() { return potentialEnergy() + kineticEnergy(); }

        T calculateTimestep();

        // Getters and setters
        void pushBody(const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity, const T bodyMass);
        void popBody(const size_t bodyID);

        const std::pair<Maths::Vec3<T>, Maths::Vec3<T>> getStateVector(const size_t bodyID);
        bool setStateVector(const size_t bodyID, const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity) {
            if (bodyID < positionX.size()) return false;

            return true;
        }

        // Getters
        const T x(const size_t bodyID);
        const T y(const size_t bodyID);
        const T z(const size_t bodyID);

        const T vX(const size_t bodyID);
        const T vY(const size_t bodyID);
        const T vZ(const size_t bodyID);

        // Setters
    private:
        TimeStep stepType;
        T timestep;

        Integrator integrator;

        Maths::Vec<T> positionX, positionY, positionZ;
        Maths::Vec<T> velocityX, velocityY, velocityZ;
        Maths::Vec<T> accelX, accelY, accelZ;
        Maths::Vec<T> mass;
    };


}// namespace Arktos::Physics
