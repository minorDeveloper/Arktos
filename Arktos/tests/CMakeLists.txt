find_package(Corrade REQUIRED TestSuite)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

enable_testing()
corrade_add_test(VecTest Arktos/Maths/VecTest.cpp LIBRARIES Arktos_Lib)