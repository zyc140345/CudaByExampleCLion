add_executable(add_loop_cpu add_loop_cpu.cu)
add_executable(add_loop_gpu add_loop_gpu.cu)
add_executable(add_loop_long add_loop_long.cu)

add_executable(julia_cpu julia_cpu.cu)
target_link_libraries(julia_cpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(julia_gpu julia_gpu.cu)
target_link_libraries(julia_gpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(burning_ship_gpu burning_ship_gpu.cu)
target_link_libraries(burning_ship_gpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})