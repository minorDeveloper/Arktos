find_package(Corrade REQUIRED TestSuite)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

enable_testing()

corrade_add_test(BaseVecTest Arktos/Maths/BaseVecTest.cpp LIBRARIES Arktos_Lib)
corrade_add_test(VecTest Arktos/Maths/VecTest.cpp LIBRARIES Arktos_Lib)
corrade_add_test(Vec3Test Arktos/Maths/Vec3Test.cpp LIBRARIES Arktos_Lib)

corrade_add_test(NbodyTest Arktos/Physics/NbodyTest.cpp LIBRARIES Arktos_Lib)
corrade_add_test(SystemTest Arktos/Physics/SystemTest.cpp LIBRARIES Arktos_Lib)
