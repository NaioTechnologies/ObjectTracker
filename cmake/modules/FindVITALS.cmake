#-------------------------------------------------------------------------------
#
#   LIBVITALS Module finder.
#
#   LIBVITALS_FOUND           - system has libvitals
#   LIBVITALS_INCLUDE_DIRS    - the libvitals include directory
#   LIBVITALS_LIBRARIES       - Link these to use libvitals
#   LIBVITALS_DEFINITIONS     - Compiler switches required for using libvitals
#   LIBVITALS_VERSION         - libvitals version
#
#-------------------------------------------------------------------------------

if( VITALS_LIBRARIES AND VITALS_INCLUDE_DIRS )
    set( VITALS_FOUND true )
else( VITALS_LIBRARIES AND VITALS_INCLUDE_DIRS )

    # Looking for include folders
    find_path( VITALS_INCLUDE_DIR
        NAMES
            BuildVersion.h
        PATHS
            $ENV{LIB_DIR}/include
            /usr/include
            /usr/local/include
        PATH_SUFFIXES
            vitals
    )
    SET( VITALS_INCLUDE_DIRS ${VITALS_INCLUDE_DIR} )

    # Looking for library folders
    find_library( VITALS_LIBRARY
        NAMES
            libvitals.so
        HINTS
            $ENV{LIB_DIR}/lib
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
            /lib/i386-linux-gnu
    )
    SET( VITALS_LIBRARIES ${VITALS_LIBRARY} )

    # Checking default library version
    if( NOT VITALS_FIND_VERSION )
        if( NOT VITALS_FIND_VERSION_MAJOR )
            set( VITALS_FIND_VERSION_MAJOR 0 )
        endif( NOT VITALS_FIND_VERSION_MAJOR )
        if( NOT VITALS_FIND_VERSION_MINOR )
            set( VITALS_FIND_VERSION_MINOR 4 )
        endif( NOT VITALS_FIND_VERSION_MINOR )
        if( NOT VITALS_FIND_VERSION_PATCH )
            set( VITALS_FIND_VERSION_PATCH 0 )
        endif( NOT VITALS_FIND_VERSION_PATCH )
        set( VITALS_FIND_VERSION "${VITALS_FIND_VERSION_MAJOR}.${VITALS_FIND_VERSION_MINOR}.${VITALS_FIND_VERSION_PATCH}" )
    endif( NOT VITALS_FIND_VERSION )

    macro( check_version )
        file( READ "${VITALS_INCLUDE_DIR}/BuildVersion.h" BUILD_VERSION_HEADER )

        # Major version
        string( REGEX MATCH "define[ \t]+LIBRARY_MAJOR_VERSION[ \t]+[\"]([0-9]+)+[\"]" VITALS_MAJOR_VERSION_MATCH "${BUILD_VERSION_HEADER}" )
        set( VITALS_MAJOR_VERSION "${CMAKE_MATCH_1}" )

        # Minor version
        string( REGEX MATCH "define[ \t]+LIBRARY_MINOR_VERSION[ \t]+[\"]([0-9]+)+[\"]" VITALS_MINOR_VERSION_MATCH "${BUILD_VERSION_HEADER}" )
        set( VITALS_MINOR_VERSION "${CMAKE_MATCH_1}" )

        # Patch version
        string( REGEX MATCH "define[ \t]+LIBRARY_PATCH_VERSION[ \t]+[\"]([0-9]+)+[\"]" VITALS_PATCH_VERSION_MATCH "${BUILD_VERSION_HEADER}" )
        set( VITALS_PATCH_VERSION "${CMAKE_MATCH_1}" )

        set( VITALS_VERSION ${VITALS_MAJOR_VERSION}.${VITALS_MINOR_VERSION}.${VITALS_PATCH_VERSION} )

        if( ${VITALS_VERSION} VERSION_LESS ${VITALS_FIND_VERSION} )
            set( VITALS_VERSION_OK FALSE )
        else( ${VITALS_VERSION} VERSION_LESS ${VITALS_FIND_VERSION} )
            set( VITALS_VERSION_OK TRUE )
        endif( ${VITALS_VERSION} VERSION_LESS ${VITALS_FIND_VERSION} )

        if( NOT VITALS_VERSION_OK )
            message( STATUS "Vitals version \"${VITALS_VERSION}\" found in ${VITALS_INCLUDE_DIR}, "
                         "but at least version \"${VITALS_FIND_VERSION}\" is required" )
        endif( NOT VITALS_VERSION_OK )
    endmacro( check_version )

    check_version()

    if( VITALS_INCLUDE_DIRS AND VITALS_LIBRARIES AND VITALS_VERSION_OK )
        set( VITALS_FOUND true )
    endif( VITALS_INCLUDE_DIRS AND VITALS_LIBRARIES AND VITALS_VERSION_OK )

    if( VITALS_FOUND )
        if( NOT VITALS_FIND_QUIETLY )
            message( STATUS "Found Vitals (Required is at least version \"${VITALS_VERSION}\", found version \"${VITALS_FIND_VERSION}\")" )
            message( STATUS " - Includes: ${VITALS_INCLUDE_DIRS}" )
            message( STATUS " - Libraries: ${VITALS_LIBRARIES}" )
        endif( NOT VITALS_FIND_QUIETLY )
    else( VITALS_FOUND )
        if( VITALS_FIND_REQUIRED )
            message( FATAL_ERROR "Could not find Vitals" )
        endif( VITALS_FIND_REQUIRED )
    endif( VITALS_FOUND )

    mark_as_advanced( VITALS_INCLUDE_DIRS VITALS_LIBRARIES )

endif( VITALS_LIBRARIES AND VITALS_INCLUDE_DIRS )
