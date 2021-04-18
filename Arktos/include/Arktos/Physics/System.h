#pragma once

#include <math.h>
#include <iostream>
#include <fstream>

#include "Arktos/Maths/Vec.h"
#include "Arktos/Maths/Vec3.h"

#include "Arktos/Physics/Constants.h"

// TODO: Document
// TODO: Test

namespace Arktos::Physics {
    enum class TimeStep { Fixed, Variable };
    enum class Integrator { Direct, RK4 };
    enum class OutputMode { None, StartEnd, Full};

    struct SystemParameters{
        TimeStep   timeStep;
        Integrator integrator;
        OutputMode outputMode;

        SystemParameters(const TimeStep timeStep, const Integrator integrator, const OutputMode outputMode) noexcept:
            timeStep{timeStep}, integrator{integrator}, outputMode{outputMode} {};
    };

    template<class T>
    class System {
    public:
        System() noexcept:  positionX{}, positionY{}, positionZ{},
                            velocityX{}, velocityY{}, velocityZ{},
                            accelX{}, accelY{}, accelZ{}, mass{},
                            bodies{0}, timeStep{0}, time{0}, totalMass{0} { fileOutput.open("SystemFile.dat"); };

        System(const size_t bodies, const SystemParameters& systemParameters) noexcept:  positionX{bodies}, positionY{bodies}, positionZ{bodies},
                                                                        velocityX{bodies}, velocityY{bodies}, velocityZ{bodies},
                                                                        accelX{bodies}, accelY{bodies}, accelZ{bodies},
                                                                        mass{bodies}, bodies{bodies}, timeStep{0}, time{0}, totalMass{0},
                                                                        systemParameters{systemParameters} { fileOutput.open("SystemFile.dat"); };

        ~System() { fileOutput.close(); };
        // Frame shifting
        const Maths::Vec3<T> centreOfMass();
        const Maths::Vec3<T> centreOfVelocity();
        void toCoMoving();

        // Energy conservation
        const T potentialEnergy();
        const T kineticEnergy();
        const T totalEnergy() { return potentialEnergy() + kineticEnergy(); }

        T calculateTimestep();

        // Getters and setters
        void pushBody(const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity, const T bodyMass);
        void popBody(const size_t bodyID);

        const std::pair<Maths::Vec3<T>, Maths::Vec3<T>> getStateVector(size_t bodyID);

        bool setStateVector(size_t bodyID, const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity, const T mass_);
        bool setStateVector(size_t bodyID, const T x, const T y, const T z, const T vX, const T vY, const T vZ, const T mass_);

        void outputState();


        void advanceUntilTime(double endTime);
        void advanceSteps(size_t n);

        void setTimeStep(T timeStep_) {
            timeStep = timeStep_;
        }
    private:
        std::ofstream fileOutput;
        const SystemParameters systemParameters;
        T timeStep;
        T time;


        size_t bodies;
        Maths::Vec<T> positionX, positionY, positionZ;
        Maths::Vec<T> velocityX, velocityY, velocityZ;
        Maths::Vec<T> accelX, accelY, accelZ;
        Maths::Vec<T> mass;
        T totalMass = (T)0;

        void updateTimestep();
        T variableTimestep();

        void advance();
        void updateAcceleration(); // TODO: Make this a virtual function in the future so System becomes an abstract base class

        void updateMass(const size_t bodyID, const T mass_);
    };

    template<class T>
    bool System<T>::setStateVector(const size_t bodyID, const Maths::Vec3<T>& position, const Maths::Vec3<T>& velocity, const T mass_) {
        if (bodyID > this->bodies) return false;

        positionX[bodyID] = position.x();
        positionY[bodyID] = position.y();
        positionZ[bodyID] = position.z();

        velocityX[bodyID] = velocity.x();
        velocityY[bodyID] = velocity.y();
        velocityZ[bodyID] = velocity.z();

        mass[bodyID] = mass_;

        return true;
    }

    template<class T>
    bool System<T>::setStateVector(size_t bodyID, const T x, const T y, const T z, const T vX, const T vY, const T vZ, const T mass_) {
        if (bodyID > this->bodies) return false;

        positionX[bodyID] = x;
        positionY[bodyID] = y;
        positionZ[bodyID] = z;

        velocityX[bodyID] = vX;
        velocityY[bodyID] = vY;
        velocityZ[bodyID] = vZ;

        updateMass(bodyID, mass_);

        return true;
    }

    template<class T>
    void System<T>::advanceUntilTime(double endTime) {
        // Calculate initial time-step
        updateTimestep();

        do {
            advance();
            // We calculate the time step after advancing in the loop to ensure that we can never exceed the endTime
            // (which could happen if the time-step suddenly increases for the final advance)
            updateTimestep();
        }
        while (time + timeStep < endTime);

        timeStep = endTime - time;
        advance();
    }

    template<class T>
    void System<T>::advanceSteps(size_t n) {
        for (size_t i = 0; i < n; i++)
        {
            updateTimestep();
            advance();
        }
    }

    template<class T>
    void System<T>::advance() {
        switch (systemParameters.outputMode) {
            case OutputMode::None:
                break;
            case OutputMode::StartEnd:
                break;
            case OutputMode::Full:
                fileOutput << time << "    ";
                for (size_t i = 0; i < bodies; i++) {
                    fileOutput << positionX[i] << "    " << positionY[i] << " ";
                }
                fileOutput << std::endl;
                break;
        }

        switch (systemParameters.integrator) {
            case Integrator::Direct:
                updateAcceleration();

                positionX += velocityX * timeStep;
                positionY += velocityY * timeStep;
                positionZ += velocityZ * timeStep;

                velocityX += accelX * timeStep;
                velocityY += accelY * timeStep;
                velocityZ += accelZ * timeStep;

                break;
            case Integrator::RK4:
                break;
        }

        time += timeStep;
    }

    template<class T>
    void System<T>::updateTimestep() {
        if (systemParameters.timeStep == TimeStep::Fixed) return;
        // TODO: Add support for variable timesteps
    }

    template<class T>
    void System<T>::updateAcceleration() {
        Arktos::Maths::Vec3<T> acceleration;
        for (size_t i = 0; i < bodies; i++) {
            acceleration = {0.0, 0.0, 0.0};

            for (size_t j = 0; j < bodies; j++) {
                if (i == j) continue;

                // Calculate the separation vector
                Arktos::Maths::Vec3<T> separation = {positionX[j] - positionX[i],
                                                     positionY[j] - positionY[i],
                                                     positionZ[j] - positionZ[i]};

                T GMSep = Arktos::Physics::Constants::grav * mass[j] / pow(separation.length(), 3);

                acceleration += GMSep * separation;
            }
            // Update the acceleration
            accelX[i] = acceleration.x();
            accelY[i] = acceleration.y();
            accelZ[i] = acceleration.z();
        }
    }

    template<class T>
    const Maths::Vec3<T> System<T>::centreOfMass() {
        T massFraction;
        Maths::Vec3<T> com;
        Maths::Vec3<T> temp;

        totalMass = 0;
        for (size_t i = 0; i < bodies; i++) {
            totalMass += mass[i];
        }

        for (size_t i = 0; i < bodies; i++) {
            massFraction = mass[i] / totalMass;
            temp = { positionX[i], positionY[i], positionZ[i] };
            com += massFraction * temp;
        }

        return com;
    }

    template<class T>
    const Maths::Vec3<T> System<T>::centreOfVelocity() {
        T massFraction;
        Maths::Vec3<T> cov;
        Maths::Vec3<T> temp;

        // FixMe: The solution for keeping track of the total mass so we don't have to do this isn't working yet
        totalMass = 0;
        for (size_t i = 0; i < bodies; i++) {
            totalMass += mass[i];
        }

        for (size_t i = 0; i < bodies; i++) {
            massFraction = mass[i] / totalMass;
            temp = { velocityX[i], velocityY[i], velocityZ[i] };
            cov += massFraction * temp;
        }

        return cov;
    }

    template<class T>
    void System<T>::toCoMoving() {
        // FixMe: Something about this isn't quite right yet. I'm not sure why.
        Maths::Vec3<T> com = centreOfMass();
        Maths::Vec3<T> cov = centreOfVelocity();

        positionX -= com.x();
        positionY -= com.y();
        positionZ -= com.z();

        velocityX -= cov.x();
        velocityY -= cov.y();
        velocityZ -= cov.z();
    }

    template<class T>
    void System<T>::updateMass(const size_t bodyID, const T mass_) {
        totalMass -= mass[bodyID];
        mass[bodyID] = mass_;
        totalMass += mass[bodyID];
    }


}// namespace Arktos::Physics
