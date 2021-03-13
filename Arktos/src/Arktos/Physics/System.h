#pragma once

#include <vector>

namespace Arktos::Physics {
    template<class T>
    class System {
    private:
        T timestep;
        std::vector<T> positionX, positionY, positionZ;
        std::vector<T> velocityX, velocityY, velocityZ;
        std::vector<T> accelX, accelY, accelZ;
        const std::vector<T> mass;

    public:
        ~System();
        System();
        void addBody();
    };
}// namespace Arktos::Physics
