#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define MAX_ITERATION 5000
#define THRESHOLD 1e-5

// Global Variables used across different functions
int global_num_points;
int global_num_threads;
int global_num_iter;
int global_centroids;
int *global_data_points;
float *global_iter_centroids;
int *global_data_point_cluster;
int **global_iter_cluster_count;

// Defined global delta
double global_deta = THRESHOLD + 1;

void readData(const char *dataset_filename, int *N, int **data_points){
    FILE *f = fopen(dataset_filename, "r");
    fscanf(f, "%d", N);
    *data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
    for (int i = 0; i < (*N) * 3; i++){
        fscanf(f, "%d", (*data_points + i));
    }
    fclose(f);
}

void writeCluster(const char *cluster_filename, int N, int *cluster_points){
    FILE *f = fopen(cluster_filename, "w");
    for (int i = 0; i < N; i++)
    {
        fprintf(f, "%d %d %d %d\n",
                *(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
                *(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
    }
    fclose(f);
}

void writeCentroids(const char *centroid_filename, int centroids, int num_iterations, float *centroids_iteration){
    FILE *f = fopen(centroid_filename, "w");
    for (int i = 0; i < num_iterations + 1; i++){
        for (int j = 0; j < centroids; j++){
            fprintf(f, "%f %f %f, ",
                    *(centroids_iteration + (i * centroids + j) * 3),        //x coordinate
                    *(centroids_iteration + (i * centroids + j) * 3 + 1),  //y coordinate
                    *(centroids_iteration + (i * centroids + j) * 3 + 2)); //z coordinate
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void calculateKmeansInThread(int *tid){
    int *id = (int*)tid;

    // Assigning data points range to each thread
    int data_length_per_thread = global_num_points / global_num_threads;
    int start = (*id) * data_length_per_thread;
    int end = start + data_length_per_thread;
    if (end + data_length_per_thread > global_num_points)
    {
        //To assign last undistributed points to this thread for computation, change end index to global_num_points
        end = global_num_points;
        data_length_per_thread = global_num_points - start;
    }

    // printf("Thread ID:%d, start:%d, end:%d\n", *id, start, end);

    double min_distance, current_distance;

    // Cluster id associated with each point
    int *current_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));

    // Cluster location or centroid (x,y,z) coordinates for centroids clusters in a iteration
    float *cluster_points_sum = (float *)malloc(global_centroids * 3 * sizeof(float));

    // No. of points in a cluster for a iteration
    int *current_cluster_count = (int *)malloc(global_centroids * sizeof(int));

    // Start of loop
    int iter_counter = 0;
    while ((global_deta > THRESHOLD) && (iter_counter < MAX_ITERATION)){
        // Initialize cluster_points_sum or centroid to 0.0
        for (int i = 0; i < global_centroids * 3; i++)
            cluster_points_sum[i] = 0.0;

        // Initialize number of points for each cluster to 0
        for (int i = 0; i < global_centroids; i++)
            current_cluster_count[i] = 0;

        for (int i = start; i < end; i++){
            //Assign these points to their nearest cluster
            min_distance = DBL_MAX;
            for (int j = 0; j < global_centroids; j++){
                current_distance = pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3] - (float)global_data_points[i * 3]), 2.0) +
                               pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3 + 1] - (float)global_data_points[i * 3 + 1]), 2.0) +
                               pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3 + 2] - (float)global_data_points[i * 3 + 2]), 2.0);
                if (min_distance > current_distance){
                    min_distance = current_distance;
                    current_cluster_id[i - start] = j;
                }
            }

            //Update local count of number of points inside cluster
            current_cluster_count[current_cluster_id[i - start]] += 1;

            // Update local sum of cluster data points
            cluster_points_sum[current_cluster_id[i - start] * 3] += (float)global_data_points[i * 3];
            cluster_points_sum[current_cluster_id[i - start] * 3 + 1] += (float)global_data_points[i * 3 + 1];
            cluster_points_sum[current_cluster_id[i - start] * 3 + 2] += (float)global_data_points[i * 3 + 2];
        }


#pragma omp critical
        {
            for (int i = 0; i < global_centroids; i++){
                if (current_cluster_count[i] == 0){
                    continue;
                }
                global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] = (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] * global_iter_cluster_count[iter_counter][i] + cluster_points_sum[i * 3]) / (float)(global_iter_cluster_count[iter_counter][i] + current_cluster_count[i]);
                global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] = (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] * global_iter_cluster_count[iter_counter][i] + cluster_points_sum[i * 3 + 1]) / (float)(global_iter_cluster_count[iter_counter][i] + current_cluster_count[i]);
                global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] = (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] * global_iter_cluster_count[iter_counter][i] + cluster_points_sum[i * 3 + 2]) / (float)(global_iter_cluster_count[iter_counter][i] + current_cluster_count[i]);
                global_iter_cluster_count[iter_counter][i] += current_cluster_count[i];
            }
        }


#pragma omp barrier
        if (*id == 0){
            double temp_delta = 0.0;
            for (int i = 0; i < global_centroids; i++){
                temp_delta += 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3]) * 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3]) + 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3 + 1]) * 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3 + 1]) + 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3 + 2]) * 
                (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] - global_iter_centroids[((iter_counter) * global_centroids + i) * 3 + 2]);
            }
            global_deta = temp_delta;
            global_num_iter++;
        }


#pragma omp barrier
        iter_counter++;
    }


// Assign points to final choice of cluster centroids
    for (int i = start; i < end; i++)
    {
        // Assign points to clusters
        global_data_point_cluster[i * 4] = global_data_points[i * 3];
        global_data_point_cluster[i * 4 + 1] = global_data_points[i * 3 + 1];
        global_data_point_cluster[i * 4 + 2] = global_data_points[i * 3 + 2];
        global_data_point_cluster[i * 4 + 3] = current_cluster_id[i - start];
        assert(current_cluster_id[i - start] >= 0 && current_cluster_id[i - start] < global_centroids);
    }
}

void kmeansOpenMP(int num_threads, int N, int centroids, int *data_points, int **cluster_id, float **centroids_iteration, int *num_iterations){

    // Initialize global variables
    global_num_points = N;
    global_num_threads = num_threads;
    global_num_iter = 0;
    global_centroids = centroids;
    global_data_points = data_points;

    *cluster_id = (int *)malloc(N * 4 * sizeof(int));   //Allocating space of 4 units each for N data points
    global_data_point_cluster = *cluster_id;

   
    global_iter_centroids = (float *)calloc((MAX_ITERATION + 1) * centroids * 3, sizeof(float));

    // Assigning first centroids points to be initial centroids
    for (int i = 0; i < centroids; i++)
    {
        global_iter_centroids[i * 3] = data_points[i * 3];
        global_iter_centroids[i * 3 + 1] = data_points[i * 3 + 1];
        global_iter_centroids[i * 3 + 2] = data_points[i * 3 + 2];
    }

    printf("Inital centroids:\n");
    for(int i = 0; i < global_centroids; i++){
        printf("%f\t%f\t%f\n", global_iter_centroids[i * 3], global_iter_centroids[i * 3 + 1], global_iter_centroids[i * 3 + 2]);
    }

   
    global_iter_cluster_count = (int **)malloc(MAX_ITERATION * sizeof(int *));
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        global_iter_cluster_count[i] = (int *)calloc(centroids, sizeof(int));
    }

    // Creating threads
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        // printf("Thread: %d created!\n", ID);
        calculateKmeansInThread(&ID);
    }

    // Record num_iterations
    *num_iterations = global_num_iter;

    // Record number of iterations and store global_iter_centroids data into centroids_iteration
    int iter_centroids_size = (*num_iterations + 1) * centroids * 3;
    printf("Number of iterations :%d\n", *num_iterations);
    *centroids_iteration = (float *)calloc(iter_centroids_size, sizeof(float));
    for (int i = 0; i < iter_centroids_size; i++)
    {
        (*centroids_iteration)[i] = global_iter_centroids[i];
    }

    // Print final centroids after last iteration
    printf("Final centroids:\n");
    for(int i = 0; i < centroids; i++){
        printf("%f\t%f\t%f\n", (*centroids_iteration)[((*num_iterations) * centroids + i) * 3], (*centroids_iteration)[((*num_iterations) * centroids + i) * 3 + 1], (*centroids_iteration)[((*num_iterations) * centroids + i) * 3 + 2]);
    }

}

int main(int argc, char const *argv[]){

	int N;
	int num_threads = atoi(argv[1]);		// thread counts
    int centroids = atoi(argv[2]);			//number of clusters
    const char *dataset_filename = argv[3];
	int* data_points;						
	int* cluster_points;					
	float* centroids_iteration;				//centroids of each iteration
	int num_iterations;


	double start_time, end_time;
	double executed_time;
    printf("//====================Start====================//\n");
    printf("Number of threads: %d\n", num_threads);
	readData(dataset_filename, &N, &data_points);
    start_time = omp_get_wtime();
	kmeansOpenMP(num_threads, N, centroids, data_points, &cluster_points, &centroids_iteration, &num_iterations);
	end_time = omp_get_wtime();

    executed_time = end_time - start_time;
    printf("Executed time: %lf seconds\n", executed_time);

    // Creating output files for different threads and dataset
    char num_threads_char[3];
    snprintf(num_threads_char,10,"%d", num_threads);

    char cluster_filename[255] = "Cluster_threads_";
    strcat(cluster_filename,num_threads_char);
    strcat(cluster_filename,"_");
    strcat(cluster_filename,dataset_filename);

    char centroid_filename[255] = "Centroid_threads_";
    strcat(centroid_filename, num_threads_char);
    strcat(centroid_filename, "_");
    strcat(centroid_filename, dataset_filename);

	writeCluster (cluster_filename, N, cluster_points);
	writeCentroids (centroid_filename, centroids, num_iterations, centroids_iteration);
    
	char time_file_omp[100] = "Executed_time_openmp_threads_";
    strcat(time_file_omp, num_threads_char);
    strcat(time_file_omp, "_");
    strcat(time_file_omp, dataset_filename);

	FILE *fout = fopen(time_file_omp, "a");
	fprintf(fout, "%f\n", executed_time);
	fclose(fout);
    printf("//====================Finished====================//\n");
	return 0;
}
