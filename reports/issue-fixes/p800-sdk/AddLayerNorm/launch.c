#include <stddef.h>

extern int xpu_launch_config(int clusters, int cores, void *stream);
extern int xpu_launch_argument_set(const void *value, size_t size, size_t offset);
extern int xpu_launch_async(void *kernel);

int launch_dual_ln(void *kernel, int clusters, void *x, void *y, void *w, void *b,
                   void *out, void *res, int rows, float eps) {
    void *ptrs[] = {x, y, w, b, out, res};
    int rc = xpu_launch_config(clusters, 64, NULL);
    for (size_t i = 0; i < 6 && !rc; ++i)
        rc = xpu_launch_argument_set(&ptrs[i], sizeof(void *), i * sizeof(void *));
    if (!rc) rc = xpu_launch_argument_set(&rows, sizeof(rows), 6 * sizeof(void *));
    if (!rc) rc = xpu_launch_argument_set(&eps, sizeof(eps), 6 * sizeof(void *) + sizeof(rows));
    return rc ? rc : xpu_launch_async(kernel);
}
