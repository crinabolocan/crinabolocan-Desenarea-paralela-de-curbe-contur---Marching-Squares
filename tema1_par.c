// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

//structura pentru a trimite datele catre threaduri
struct thread_data {
    int thread_id; // id-ul threadului
    int num_threads; // numarul de threaduri
    ppm_image **image; // imaginea
    int step_x; 
    int step_y;
    unsigned char sigma;
    ppm_image **contour_map;
    unsigned char **grid;
    ppm_image *new_image; // imaginea redimensionata
    pthread_barrier_t *barrier; // bariera
};

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}


// functia f pentru a paraleliza functiile sample_grid si march si rescale_image
void *f (void *arg) {
    // mai jos se extrag datele din structura
    struct thread_data *data = (struct thread_data *) arg;
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;
    ppm_image **image = data->image;
    int step_x = data->step_x;
    int step_y = data->step_y;
    unsigned char sigma = data->sigma;
    ppm_image **contour_map = data->contour_map;
    unsigned char **grid = data->grid;
    ppm_image *new_image = data->new_image;
    pthread_barrier_t *barrier = data->barrier;

    // calculam de unde incepe si unde se termina fiecare thread
    int start_rescale = thread_id * RESCALE_X / num_threads;
    int end_rescale;
    if (thread_id == num_threads - 1) {
        end_rescale = RESCALE_X;
    } else {
        end_rescale = (thread_id + 1) * RESCALE_X / num_threads;
    }

    uint8_t sample[3];
  
    // use bicubic interpolation for scaling
    if((*image)->x > RESCALE_X && (*image)->y > RESCALE_Y) {

        for (int i = start_rescale; i < end_rescale; i++) {
            for (int j = 0; j < RESCALE_Y; j++) {
                float u = (float)i / (float)(RESCALE_X - 1);
                float v = (float)j / (float)(RESCALE_Y - 1);
                sample_bicubic((*image), u, v, sample);

                new_image->data[i * RESCALE_Y + j].red = sample[0];
                new_image->data[i * RESCALE_Y + j].green = sample[1];
                new_image->data[i * RESCALE_Y + j].blue = sample[2];

            }
        }
        // bariera pentru a ne asigura ca toate threadurile au terminat
        pthread_barrier_wait(barrier);

        // imaginea redimensionata
        *image = new_image;
        
    }

    // calculam de unde incepe si unde se termina fiecare thread dupa redimensionare
    int p = (*image)->x / step_x;
    int q = (*image)->y / step_y;
    int start = thread_id * p / num_threads;
    int end;
    if (thread_id == num_threads - 1) {
        end = p;
    } else {
        end = (thread_id + 1) * p / num_threads;
    }
   

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = (*image)->data[i * step_x * (*image)->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;
            // criteriul sigma
            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = (*image)->data[i * step_x * (*image)->y + (*image)->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    start = thread_id * q / num_threads;
    if (thread_id == num_threads - 1) {
        end = q;
    } else {
        end = (thread_id + 1) * q / num_threads;
    }

    for (int j = start; j < end; j++) {
        ppm_pixel curr_pixel = (*image)->data[((*image)->x - 1) * (*image)->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }

    // bariera pentru a ne asigura ca toate threadurile au terminat
    pthread_barrier_wait(barrier);

    start = thread_id * p / num_threads;
    if (thread_id == num_threads - 1) {
        end = p;
    } else {
        end = (thread_id + 1) * p / num_threads;
    }

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(*image, contour_map[k], i * step_x, j * step_y);
        }
    }

    pthread_exit(NULL);    
}


// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}


int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // initializari
    int num_threads = argv[3][0] - '0';
    pthread_t threads[num_threads];

    // citirea imaginii
    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;

    int p = image->x / step_x;
    int q = image->y / step_y;

    ppm_image *image2 = image;

    // se aloca dinamic o matrice pe X cu p+1 linii si q+1 coloane
    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    // se aloca dinamic o structura pentru a trimite datele catre threaduri
    struct thread_data *data = (struct thread_data *)malloc(num_threads * sizeof(struct thread_data));
    if(!data) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }


    // Initialize contour map
    ppm_image **contour_map = init_contour_map();

    // se aloca pentru imaginea redimensionata
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
        new_image->x = RESCALE_X;
        new_image->y = RESCALE_Y;

        new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }

    //bariera
    pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));

    if(pthread_barrier_init(barrier, NULL, num_threads)) {
        return -1;
    }

    int r;

    // crearea threadurilor si trimiterea datelor catre acestea
    for(int i = 0; i < num_threads; i++) {
        data[i].thread_id = i;
        data[i].num_threads = num_threads;
        data[i].image = &image;
        data[i].step_x = step_x;
        data[i].step_y = step_y;
        data[i].sigma = SIGMA;
        data[i].contour_map = contour_map;
        data[i].grid = grid;
        data[i].new_image = new_image;
        data[i].barrier = barrier;
        r = pthread_create(&threads[i], NULL, f, (void *) &data[i]);
        if(r) {
            return -1;
        }
        
    }

    // asteptarea threadurilor
    for(int i = 0; i < num_threads; i++) {
        r = pthread_join(threads[i], NULL);
        if(r) {
            return -1;
        }
    }


    // Write output
    write_ppm(image, argv[2]);
    free(new_image->data);
    free(new_image);
    if(p * step_x > RESCALE_X || q * step_y > RESCALE_Y) {
        free_resources(image2, contour_map, grid, step_x);

    }
    else {
        free_resources(image, contour_map, grid, step_x);
    }
    pthread_barrier_destroy(barrier);
    free(barrier);
    //dezaloca new_image



    //free(new_image->data);
    // free(new_image);
    free(data);



    return 0;
}
