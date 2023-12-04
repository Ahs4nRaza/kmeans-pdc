#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define MAX_ITER 1000
#define THRESHOLD 1e-5

int global_num_points;
int global_num_iter;
double global_deta = THRESHOLD + 1;
int global_centroids;
int *global_data_points;
float *global_iter_centroids;
int *global_data_point_cluster;


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


void find_kmeans(int N, int centroids, int* data_points, int** data_point_cluster_id, float** centroids_iteration, int* num_iterations){
    global_num_points = N;
    global_num_iter = *num_iterations;
    global_centroids = centroids;
    global_data_points = data_points;

    *data_point_cluster_id = (int *)malloc(N * 4 * sizeof(int));
    global_data_point_cluster = *data_point_cluster_id;
    global_iter_centroids = (float *)calloc((MAX_ITER + 1) * centroids * 3, sizeof(float));


    for (int i = 0; i < centroids; i++){
        global_iter_centroids[i * 3] = data_points[i * 3];
        global_iter_centroids[i * 3 + 1] = data_points[i * 3 + 1];
        global_iter_centroids[i * 3 + 2] = data_points[i * 3 + 2];
    }

    printf("Inital centroids:\n");
    for(int i = 0; i < global_centroids; i++){
        printf("%f\t%f\t%f\n", global_iter_centroids[i * 3], global_iter_centroids[i * 3 + 1], global_iter_centroids[i * 3 + 2]);
    }

    
    double min_distance, current_distance;

    // Cluster id associated with each point
    int *point_to_cluster_id = (int *)malloc(global_num_points * sizeof(int));
    float *cluster_points_sum = (float *)malloc(global_centroids * 3 * sizeof(float));
    int *points_inside_cluster_count = (int *)malloc(global_centroids * sizeof(int));

    int iter_counter = 0;
    double temp_delta = 0.0;
    while ((global_deta > THRESHOLD) && (iter_counter < MAX_ITER)){
        // Initialize cluster_points_sum or centroid to 0.0
        for (int i = 0; i < global_centroids * 3; i++)
            cluster_points_sum[i] = 0.0;

        // Initialize number of points for each cluster to 0
        for (int i = 0; i < global_centroids; i++)
            points_inside_cluster_count[i] = 0;

        for (int i = 0; i < global_num_points; i++){
            //Assign these points to their nearest cluster
            min_distance = DBL_MAX;
            for (int j = 0; j < global_centroids; j++){
                current_distance = pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3] - (float)global_data_points[i * 3]), 2.0) +
                               pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3 + 1] - (float)global_data_points[i * 3 + 1]), 2.0) +
                               pow((double)(global_iter_centroids[(iter_counter * global_centroids + j) * 3 + 2] - (float)global_data_points[i * 3 + 2]), 2.0);
                if (min_distance > current_distance){
                    min_distance = current_distance;
                    point_to_cluster_id[i] = j;
                }
            }

            points_inside_cluster_count[point_to_cluster_id[i]] += 1;

            cluster_points_sum[point_to_cluster_id[i] * 3] += (float)global_data_points[i * 3];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 1] += (float)global_data_points[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 2] += (float)global_data_points[i * 3 + 2];
        }

        for (int i = 0; i < global_centroids; i++){
            assert(points_inside_cluster_count[i] != 0);
            global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] = cluster_points_sum[i * 3] / points_inside_cluster_count[i];
            global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] = cluster_points_sum[i * 3 + 1] / points_inside_cluster_count[i];
            global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] = cluster_points_sum[i * 3 + 2] / points_inside_cluster_count[i];
        }

   
        temp_delta = 0.0;
        for (int i = 0; i < global_centroids; i++){
            temp_delta += (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3]) * (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3]) + (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3 + 1]) * (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 1] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3 + 1]) + (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3 + 2]) * (global_iter_centroids[((iter_counter + 1) * global_centroids + i) * 3 + 2] - global_iter_centroids[((iter_counter)*global_centroids + i) * 3 + 2]);
        }
        global_deta = temp_delta;

        iter_counter++;
    }

   
    global_num_iter = iter_counter;

    for (int i = 0; i < global_num_points; i++){
        // Assign points to clusters
        global_data_point_cluster[i * 4] = global_data_points[i * 3];
        global_data_point_cluster[i * 4 + 1] = global_data_points[i * 3 + 1];
        global_data_point_cluster[i * 4 + 2] = global_data_points[i * 3 + 2];
        global_data_point_cluster[i * 4 + 3] = point_to_cluster_id[i];
        assert(point_to_cluster_id[i] >= 0 && point_to_cluster_id[i] < global_centroids);
    }


    // Record number of iterations and store global_iter_centroids data into centroids_iteration
    *num_iterations = global_num_iter;
    int centroids_size = (*num_iterations + 1) * centroids * 3;
    printf("number of iterations:%d\n", global_num_iter);
    *centroids_iteration = (float *)calloc(centroids_size, sizeof(float));

    for (int i = 0; i < centroids_size; i++){
        (*centroids_iteration)[i] = global_iter_centroids[i];
    }

    // Print final centroids after last iteration
    printf("Final centroids:\n");
    for(int i = 0; i < centroids; i++){
        printf("%f\t%f\t%f\n", (*centroids_iteration)[((*num_iterations) * centroids + i) * 3], (*centroids_iteration)[((*num_iterations) * centroids + i) * 3 + 1], (*centroids_iteration)[((*num_iterations) * centroids + i) * 3 + 2]);
    }
}

int main(int argc, char const *argv[]){

	//---------------------------------------------------------------------//
	int N;					
	int centroids = atoi(argv[1]);          //number of clusters
    const char *dataset_filename = argv[2];
	int* data_points;		//Data points (input)
	int* cluster_points;	
	float* centroids_iteration;		
	int num_iterations;     
    double start_time, end_time;
    double executed_time;
	//---------------------------------------------------------------------//


	readData (dataset_filename, &N, &data_points);
    printf("//====================Sequential K-Means Started!====================//\n");
	start_time = (double)clock() / (double)CLOCKS_PER_SEC;
	find_kmeans(N, centroids, data_points, &cluster_points, &centroids_iteration, &num_iterations);
	end_time = (double)clock() / (double)CLOCKS_PER_SEC;

    executed_time = end_time - start_time;
    printf("Executed Time: %lf \n", executed_time);
    printf("//====================Finished!====================//\n");

    char cluster_filename[255] = "cluster_output_";
    strcat(cluster_filename, dataset_filename);

    char centroid_filename[255] = "centroid_output_dataset";
    strcat(centroid_filename, dataset_filename);

	writeCluster(cluster_filename, N, cluster_points);
	writeCentroids(centroid_filename, centroids, num_iterations, centroids_iteration);
    
	char time_file[100] = "executed_time_sequential_dataset";
    strcat(time_file, dataset_filename);

	FILE *fout = fopen(time_file, "a");
	fprintf(fout, "%f\n", executed_time);
	fclose(fout);
	return 0;
}
