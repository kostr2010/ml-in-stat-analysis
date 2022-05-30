#include <stdlib.h>
#include <string.h>

int main()
{
    int* a = malloc(10 * sizeof(int));

    free(a);

    a[5] = 10;

    return 0;
}