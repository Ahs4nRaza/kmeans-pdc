#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);

    if (N <= 0) {
        printf("Please provide a valid positive integer for N.\n");
        return 1;
    }

    char filename[10];
    sprintf(filename, "%d.txt", N);
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    fprintf(file, "%d\n", N);

    srand(time(NULL));

    for (int i = 0; i < N; ++i) {
        int x = rand() % 2001 - 1000;
        int y = rand() % 2001 - 1000;
        int z = rand() % 2001 - 1000;
        fprintf(file, "%d %d %d\n", x, y, z);
    }

    fclose(file);

    printf("Successfully generated %d random sets of coordinates and stored them in %s.\n", N, filename);

    return 0;
}

