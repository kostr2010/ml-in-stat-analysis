// #include "exp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int FIVE = 5;
int TRUE = 1;

int* data = NULL;

void a()
{
    data = (int*)calloc(100, sizeof(int));
    if (FIVE == 5) {
        free(data);
    }
}

void fopen_good()
{
    FILE* fdata;
    fdata = NULL;
    fdata = fopen("file.txt", "r");
    if (TRUE) {
        if (fdata != NULL) {
            fclose(fdata);
        }
    }
}

int main()
{
    a();
    fopen_good();
    free(data);

    return 0;
}

// tool for SA devs to automatically make rules set 
// 1) small dataset
// 2) why small
// 3) TODO: make dataset, hoe to and why needed
// 4) 