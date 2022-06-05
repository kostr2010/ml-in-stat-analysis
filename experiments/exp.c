// #include "exp.h"
#include <stdlib.h>
#include <string.h>

static const int FIVE = 5;

int* data = NULL;

void a()
{
    data = (int*)calloc(100, sizeof(int));
    if (FIVE != 5) {
        free(data);
    }
}

int main()
{
    a();

    free(data);

    return 0;
}