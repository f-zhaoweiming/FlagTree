#include <torch/csrc/autograd/python_variable.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

extern "C" int launch_dual_ln(void *, int, void *, void *, void *, void *, void *, void *, int, float);

static PyObject *invalid(const char *message) {
    PyErr_SetString(PyExc_ValueError, message);
    return nullptr;
}

static PyObject *checked_launch(PyObject *, PyObject *args) {
    PyObject *capsule, *objects[6];
    int cols, rows_per_cluster, device;
    unsigned long long stream;
    double eps;
    if (!PyArg_ParseTuple(args, "OiiiKdOOOOOO", &capsule, &cols, &rows_per_cluster, &device, &stream, &eps,
                          &objects[0], &objects[1], &objects[2], &objects[3], &objects[4], &objects[5]))
        return nullptr;
    void *descriptor = PyCapsule_GetPointer(capsule, "sdk04.cluster");
    if (!descriptor) return nullptr;
    if (cols != 64 && cols != 1024 && cols != 2048) return invalid("unsupported width");
    if (rows_per_cluster != 16 && rows_per_cluster != 32 && rows_per_cluster != 64) return invalid("invalid launch geometry");
    if (stream || !std::isfinite(eps) || eps <= 0)
        return invalid("expected the active default stream and a positive finite epsilon");
    try {
        int64_t rows = 0;
        uintptr_t ptr[6], bytes[6];
        for (int i = 0; i < 6; ++i) {
            if (!THPVariable_CheckExact(objects[i])) Py_RETURN_NOTIMPLEMENTED;
            const auto &t = THPVariable_Unpack(objects[i]);
            if (!t.is_cuda() || t.get_device() != device || t.scalar_type() != at::kHalf ||
                t.layout() != at::kStrided || !t.is_contiguous() || t.requires_grad() || t.is_neg() || t.is_conj())
                return invalid("expected contiguous FP16 inference tensors on the active device");
            if (i == 0) {
                if (t.dim() != 2) return invalid("expected a matrix");
                rows = t.size(0);
                if (rows <= 0 || rows >= (int64_t{1} << 31) / cols - 64)
                    return invalid("unsupported row count");
            }
            if (i == 2 || i == 3) {
                if (t.dim() != 1 || t.size(0) != cols) return invalid("affine shape mismatch");
            } else if (t.dim() != 2 || t.size(0) != rows || t.size(1) != cols) {
                return invalid("matrix shape mismatch");
            }
            ptr[i] = reinterpret_cast<uintptr_t>(t.data_ptr());
            bytes[i] = t.numel() * sizeof(uint16_t);
            if (ptr[i] % 64) return invalid("KL3 buffers require 64-byte alignment");
            if (i >= 4)
                for (int j = 0; j < i; ++j)
                    if (ptr[i] < ptr[j] + bytes[j] && ptr[j] < ptr[i] + bytes[i])
                        return invalid("output overlaps another buffer");
        }
        int clusters = std::min<int64_t>(12, (rows+rows_per_cluster-1)/rows_per_cluster);
        int rc = launch_dual_ln(descriptor, clusters, (void*)ptr[0], (void*)ptr[1], (void*)ptr[2],
                                (void*)ptr[3], (void*)ptr[4], (void*)ptr[5], rows, eps);
        if (rc) return PyErr_Format(PyExc_RuntimeError, "KL3 launch failed with status %d", rc);
        return PyTuple_Pack(2, objects[4], objects[5]);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
static PyMethodDef methods[] = {{"launch", checked_launch, METH_VARARGS, nullptr}, {nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_sdk04_checked", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__sdk04_checked() { return PyModule_Create(&module); }
