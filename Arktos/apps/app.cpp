

#include <Corrade/Utility/DebugStl.h>

#include <Magnum/Math/Vector.h>

#include "Arktos/Maths/Vec.h"
#include "Arktos/Physics/System.h"
#include "Arktos/Physics/Constants.h"

using namespace Arktos::Physics;

int main() {
    // Setup the system
    SystemParameters parameters(TimeStep::Fixed, Integrator::Direct, OutputMode::Full);
    System<Magnum::Double> system(size_t(2), parameters);
    double timeStep = 24.0 * 3600.0 / 10.0;
    system.setTimeStep(timeStep);

    // Maybe provide it with different physical constants
    system.setStateVector(size_t(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Constants::Msol);
    system.setStateVector(size_t(1), Constants::AU, 0.0, 0.0, 0.0, 29.78 * Constants::km, 0.0, Constants::Mearth);
    system.toCoMoving();

    // Tell it to iterate until a certain number of steps
    system.advanceSteps(365.25 * 24.0 * 3600 / timeStep);

    Corrade::Utility::Debug{} << "test" << "this is a now";

    return 0;
}


//#include "Arktos.h"
//MAGNUM_APPLICATION_MAIN(Arktos::BaseApplication)