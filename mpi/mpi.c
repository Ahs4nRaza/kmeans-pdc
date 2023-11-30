#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <malloc.h>

double getCpuTime(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

// Calculate distance between 2 points
float calculateDistance(const float* point1, const float* point2, const int size) {
    float dist = 0.0;
    for (int i = 0; i < size; i++) {
        float diff = point2[i] - point1[i];
        dist += diff * diff;
    }
    return dist;
}

void readData(const char* filename, int* N, float** data_points) {

	FILE* f = fopen(filename, "r");
	if (f == NULL) {
		return;
	}
	fscanf(f, "%d", N);
	//printf("Number of data points in the dataset: %d\n", *N);
	*data_points = (float*)malloc(((*N) * 3) * sizeof(float));

	for (int i = 0; i < (*N) * 3; i++) {
		fscanf(f, "%f", (*data_points + i));
	}
	fclose(f);
}

void writeCluster(const char* filename, int N, float* cluster_points) {

	FILE* f = fopen(filename, "w");

	for (int i = 0; i < N; i++) {
		fprintf(f, "%f %f %f %d\n",
			*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
			*(cluster_points + (i * 4) + 2), (int)*(cluster_points + (i * 4) + 3));
	}
	fclose(f);
}

void writeCentroids(const char* filename, int K, float* centroids, int iteration) {
    FILE* f;
    if (iteration == 0) {
        f = fopen(filename, "w");  // Open the file in write mode for the first iteration
    } else {
        f = fopen(filename, "a");  // Open the file in append mode for subsequent iterations
    }

    for (int i = 0; i < K; i++) {
        fprintf(f, "%f %f %f\n", centroids[i * 3],
            centroids[i * 3 + 1],
            centroids[i * 3 + 2]);
    }
    fprintf(f, "\n");
    fclose(f);
}


int main(int argc, char** argv) {

    int num_centroids = atoi(argv[1]);  // number of clusters.
    const char* dataset_filename = argv[2]; // Data file name
    int N;
    int d;
    float* data_points;
    int num_iterations=0;
    readData(dataset_filename, &N, &data_points);
	//printf("//====================MPI K-Means Started!====================//\n");
    float* final_output_points = (float*)malloc(N * 4 * sizeof(float));
    // Initial MPI
    MPI_Init(NULL, NULL);
    int rank, num_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    double start_time = getCpuTime();
    int num_data_per_process = N / num_process;
    printf("Procs # : %d\n",num_process);
    // The data assigned to this process.
    float* data_in_process;
    data_in_process = (float*)malloc(num_data_per_process * 3 * sizeof(float));
    // The sum of data assigned to each cluster by this process.
    float* sums;
    sums = (float*)malloc(num_centroids * 3 * sizeof(float));
    // The number of data assigned to each cluster by this process. centroid integers.
    int* counts;
    counts = (int*)malloc(num_centroids * sizeof(int));
    // The current centroids against which data in process are being compared.
    // These are shipped to the process by the root process.
    float* centroids;
    centroids = (float*)malloc(num_centroids * 3 * sizeof(float));
    // The cluster assignments for each process.
    int* labels;
    labels = (int*)malloc(num_data_per_process * sizeof(int));
    int* cluster_id, iteration = 0;
    cluster_id = (int*)malloc(num_data_per_process * sizeof(float));

	char centroid_filename_mpi[255] = "centroid_output_mpi_dataset";
    strcat(centroid_filename_mpi, dataset_filename);

    float* all_data_points = NULL;
    // Sum of data assigned to each cluster by all processes.
    float* total_sums = NULL;
    // Number of data assigned to each cluster by all processes.
    int* total_counts = NULL;
    // Result of program: a cluster label for each process.
    int* all_labels;
    all_labels = (int*)malloc(num_process * num_data_per_process * sizeof(int));
    if (rank == 0) {
printf("//====================MPI K-Means Started!====================//\n");
        printf("Number of Processes: %d\n", num_process);
        all_data_points = data_points;
        for (int i = 0; i < num_centroids; i++) {
            centroids[i * 3] = all_data_points[i * 3];
            centroids[i * 3 + 1] = all_data_points[i * 3 + 1];
            centroids[i * 3 + 2] = all_data_points[i * 3 + 2];
        }
        printf("Initial Centroids:\n");
        for (int i = 0; i < num_centroids; i++) {
            printf("%f\t%f\t%f\n", centroids[i * 3], centroids[i * 3 + 1], centroids[i * 3 + 2]);
        }
        total_sums = (float*)malloc(num_centroids * 3 * sizeof(float));
        total_counts = (int*)malloc(num_centroids * sizeof(int));
    }

    // Root sends each process its share of data.
    MPI_Scatter(all_data_points, 3 * num_data_per_process, MPI_FLOAT, data_in_process, 3 * num_data_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Broadcast the current cluster centroids to all processes.
    MPI_Bcast(centroids, num_centroids * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float delta = 1.0;
    while (delta > 1e-5) {

        // Initialize cluster accumulators in each process.
        for (int i = 0; i < num_centroids * 3; i++)
            sums[i] = 0.0;
        for (int i = 0; i < num_centroids; i++)
            counts[i] = 0;
        // Find the closest centroid to each site and assign to cluster.
        float* inner_data = data_in_process;
       /* printf("num of data: %d\n", num_data_per_process);*/
        for (int i = 0; i < num_data_per_process; i++) {
            double min_distance = DBL_MAX;
            for (int j = 0; j < num_centroids; j++) {
                double current_distance = pow((double)(centroids[j * 3] - (float)inner_data[i * 3]), 2.0) +
                    pow((double)(centroids[j * 3 + 1] - (float)inner_data[i * 3 + 1]), 2.0) +
                    pow((double)(centroids[j * 3 + 2] - (float)inner_data[i * 3 + 2]), 2.0);
                if (current_distance < min_distance) {
                    min_distance = current_distance;
                    cluster_id[i] = j;
                }
            }
            counts[cluster_id[i]] += 1;
            sums[cluster_id[i] * 3] += (float)inner_data[i * 3];
            sums[cluster_id[i] * 3 + 1] += (float)inner_data[i * 3 + 1];
            sums[cluster_id[i] * 3 + 2] += (float)inner_data[i * 3 + 2];
        }

        // Gather and sum at root all cluster sums for individual processes.
        MPI_Reduce(counts, total_counts, num_centroids, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(sums, total_sums, num_centroids * 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

         if (rank == 0) {
            num_iterations++;
            for (int i = 0; i < num_centroids; i++) {
                for (int j = 0; j < 3; j++) {
                    int index = 3 * i + j;
                    total_sums[index] /= total_counts[i];
                }
            }
            for (int i = 0; i < num_centroids; i++) {
                delta = pow((double)(centroids[i * 3] - (float)total_sums[i * 3]), 2.0) +
                    pow((double)(centroids[i * 3 + 1] - (float)total_sums[i * 3 + 1]), 2.0) +
                    pow((double)(centroids[i * 3 + 2] - (float)total_sums[i * 3 + 2]), 2.0);
            }

            for (int i = 0; i < num_centroids * 3; i++) {
                centroids[i] = total_sums[i];
            }

            // Write centroids to the file after each iteration
            writeCentroids(centroid_filename_mpi, num_centroids, centroids,iteration);
            iteration++;
        }
        // Broadcast the current cluster centroids to all processes.
        MPI_Bcast(centroids, num_centroids * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // Broadcast the delta.  All processes will use this in the loop test.
        MPI_Bcast(&delta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    float* final_data = data_in_process;
    for (int i = 0; i < num_data_per_process; i++) {
        double min_distance = DBL_MAX;
        for (int j = 0; j < num_centroids; j++) {
            double current_distance = pow((double)(centroids[j * 3] - (float)final_data[i * 3]), 2.0) +
                pow((double)(centroids[j * 3 + 1] - (float)final_data[i * 3 + 1]), 2.0) +
                pow((double)(centroids[j * 3 + 2] - (float)final_data[i * 3 + 2]), 2.0);
            if (current_distance < min_distance) {
                min_distance = current_distance;
                labels[i] = j;
            }
        }
    }

    // Gather all labels into root process.
    MPI_Gather(labels, num_data_per_process, MPI_INT, all_labels, num_data_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            final_output_points[i * 4] = all_data_points[i * 3];
            final_output_points[i * 4 + 1] = all_data_points[i * 3 + 1];
            final_output_points[i * 4 + 2] = all_data_points[i * 3 + 2];
            final_output_points[i * 4 + 3] = all_labels[i];
        }
        double end_time = getCpuTime();
        printf("Final Centroids: \n");
        for (int i = 0; i < num_centroids; i++) {
            printf("%f\t%f\t%f\n", centroids[i * 3], centroids[i * 3 + 1], centroids[i * 3 + 2]);
        }
        printf("Executed time: %f seconds\n", end_time - start_time);
        printf("No. of Iteratons: %d\n",num_iterations);
printf("//====================Finished!====================//\n");
         // Create filenames for MPI code
    char cluster_filename_mpi[255] = "cluster_output_mpi_";
    strcat(cluster_filename_mpi, dataset_filename);

    
    char time_file_mpi[100] = "executed_time_mpi_dataset";
    strcat(time_file_mpi, dataset_filename);

    // Write results to files
    writeCluster(cluster_filename_mpi, N, final_output_points);
   


    FILE *fout_mpi = fopen(time_file_mpi, "a");
    fprintf(fout_mpi, "%f\n", end_time - start_time);
    fclose(fout_mpi);
    }
	  
    MPI_Finalize();
    return 0;
}
