
# - Try to find catch framework
if(Ichnea_FOUND)
  return()
endif()

find_path(Ichnea_INCLUDE_DIR include/Ichnea.h HINT ${EXTERNAL_ROOT}/extern)

set(Ichnea_INCLUDE_DIRS ${Ichnea_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Ichnea  DEFAULT_MSG Ichnea_INCLUDE_DIR)