# Copyright: (C) 2017 IIT
# Authors: Phuong Nguyen
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

#cmake_minimum_required(VERSION 2.8)
#set(skeleton2D singleperson_yarp.py)

#find_package(YARP REQUIRED)
#find_package(ICUB)

#list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
#list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})
#list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})

#add_definitions(${YARP_DEFINES})   # add yarp definitions - this contains also -D_REENTRANT
#include(YarpInstallationHelpers)   # pick up yarp's cmake scripts

#add_custom_target(singleperson_yarp_build ALL ${include_files})
#add_custom_command(TARGET singleperson_yarp_build POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${skeleton2D} ${CMAKE_BINARY_DIR}/bin
#                   COMMENT "Copying ${skeleton2D} to ${CMAKE_BINARY_DIR}/bin"
#                   DEPENDS ${include_files})

#install(PROGRAMS ${skeleton2D} DESTINATION bin PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE)
#yarp_install(PROGRAMS ${skeleton2D} DESTINATION bin PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE)
