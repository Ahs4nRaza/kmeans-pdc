# kmeans-pdc


To generate random points:
compile points_gen.c => gcc -o points points_gen.c 
execute points => ./points <no. of points> 


To run sequential program:
compile seq.c => gcc -o seq seq.c -lm
execute seq => ./seq <cluster count> <filename>

To run openmp program:
compile open.c => gcc -o open open.c -lm -fopenmp 
execute open => ./open <no.of threads> <cluster count> <filename>

To run mpi program:
compile mpi.c => mpicc -o mpi mpi.c -lm
execute mpi => mpirun -np <process count> ./mpi <cluster count> <filename>
