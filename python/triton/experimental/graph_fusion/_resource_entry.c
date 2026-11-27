/* Guarded replay for the validated CPython 3.10 P800 module ABI. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_VERSION_HEX < 0x030a0000 || PY_VERSION_HEX >= 0x030b0000
#error "This guarded entry requires CPython 3.10"
#endif

enum Field {
  MODEL, FORWARD, CODE, MODEL_DICT, HOOK_MAPS, MODE_CHECKS, CALLBACK, PROFILER,
  EVAL_FRAME, HOOK_MODULE, TENSOR_TYPE, NAMES, MANAGER, REPLAY, MISS, GUARD_MISS, REFRESH, FIELD_COUNT
};

/* These Python objects live in module state, including on interpreter teardown. */
typedef struct {
  PyObject *stance, *stance_name, *backend, *skip_guards, *default_stance;
  PyObject *profiler_flag, *global_hooks, *cache_entry, *check;
} Names;

static int visit_names(Names *n, visitproc visit, void *arg) {
  Py_VISIT(n->stance);
  Py_VISIT(n->stance_name);
  Py_VISIT(n->backend);
  Py_VISIT(n->skip_guards);
  Py_VISIT(n->default_stance);
  Py_VISIT(n->profiler_flag);
  Py_VISIT(n->global_hooks);
  Py_VISIT(n->cache_entry);
  Py_VISIT(n->check);
  return 0;
}

static int traverse(PyObject *module, visitproc visit, void *arg) {
  return visit_names((Names *)PyModule_GetState(module), visit, arg);
}

static int clear_names(PyObject *module) {
  Names *n = (Names *)PyModule_GetState(module);
  Py_CLEAR(n->stance);
  Py_CLEAR(n->stance_name);
  Py_CLEAR(n->backend);
  Py_CLEAR(n->skip_guards);
  Py_CLEAR(n->default_stance);
  Py_CLEAR(n->profiler_flag);
  Py_CLEAR(n->global_hooks);
  Py_CLEAR(n->cache_entry);
  Py_CLEAR(n->check);
  return 0;
}

static int consume_truth(PyObject *value) {
  if (value == NULL)
    return -1;
  int truth = PyObject_IsTrue(value);
  Py_DECREF(value);
  return truth;
}

static int default_stance(Names *n, PyObject *frame) {
  PyObject *stance = PyObject_GetAttr(frame, n->stance);
  if (stance == NULL)
    return -1;
  int ok = 0;
  PyObject *name = PyObject_GetAttr(stance, n->stance_name);
  if (name == NULL) {
    ok = -1;
  } else if (name == n->default_stance) {
    PyObject *backend = PyObject_GetAttr(stance, n->backend);
    if (backend == NULL) {
      ok = -1;
    } else if (backend == Py_None) {
      int skip = consume_truth(PyObject_GetAttr(stance, n->skip_guards));
      ok = skip < 0 ? -1 : !skip;
    }
    Py_XDECREF(backend);
  }
  Py_XDECREF(name);
  Py_DECREF(stance);
  return ok;
}

/* self holds the module, inputs to the preflight checks, class, and both version
 * tags. PyCFunction traverses its tuple receiver, so cycles remain collectable. */
static PyObject *try_frame_values(PyObject *self, PyObject *const *args, Py_ssize_t count) {
  PyObject *config = PyTuple_GET_ITEM(self, 1);
#define FIELD(i) PyTuple_GET_ITEM(config, (i))
#define MISS_FRAME() return Py_NewRef(FIELD(MISS))
  PyObject *names = FIELD(NAMES);
  if ((count != 2 && count != 3) || count + 1 != PyTuple_GET_SIZE(names)) {
    MISS_FRAME();
  }
  PyObject *model = FIELD(MODEL);
  PyThreadState *thread = PyThreadState_Get();
  if (thread->c_profilefunc || thread->c_tracefunc) {
    MISS_FRAME();
  }
  for (Py_ssize_t i = 0; i < count; ++i) {
    if ((PyObject *)Py_TYPE(args[i]) != FIELD(TENSOR_TYPE)) {
      MISS_FRAME();
    }
  }
  PyObject *modes = FIELD(MODE_CHECKS);
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(modes); ++i) {
    int active = consume_truth(PyObject_CallNoArgs(PyTuple_GET_ITEM(modes, i)));
    if (active < 0)
      return NULL;
    if (active) {
      MISS_FRAME();
    }
  }
  PyObject *callback = PyObject_CallNoArgs(FIELD(CALLBACK));
  if (callback == NULL)
    return NULL;
  int active = callback != Py_None;
  Py_DECREF(callback);
  if (active) {
    MISS_FRAME();
  }
  Names *n = (Names *)PyModule_GetState(PyTuple_GET_ITEM(self, 0));
  active = consume_truth(PyObject_GetAttr(FIELD(PROFILER), n->profiler_flag));
  if (active < 0)
    return NULL;
  if (active) {
    MISS_FRAME();
  }
  active = consume_truth(PyObject_CallMethodNoArgs(FIELD(HOOK_MODULE), n->global_hooks));
  if (active < 0)
    return NULL;
  if (active) {
    MISS_FRAME();
  }
  int ok = default_stance(n, FIELD(EVAL_FRAME));
  if (ok < 0)
    return NULL;
  if (!ok) {
    MISS_FRAME();
  }

  PyTypeObject *type = Py_TYPE(model);
  if ((PyObject *)type != PyTuple_GET_ITEM(self, 2) ||
      PyFunction_GET_CODE(FIELD(FORWARD)) != FIELD(CODE)) {
    MISS_FRAME();
  }
  if (!PyType_HasFeature(type, Py_TPFLAGS_VALID_VERSION_TAG) || !type->tp_version_tag ||
      type->tp_version_tag != PyLong_AsUnsignedLong(PyTuple_GET_ITEM(self, 3)))
    return Py_NewRef(FIELD(REFRESH));
  PyObject **dict = _PyObject_GetDictPtr(model);
  if (dict == NULL || *dict != FIELD(MODEL_DICT) ||
      ((PyDictObject *)*dict)->ma_version_tag !=
          PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(self, 4)))
    return Py_NewRef(FIELD(REFRESH));
  PyObject *maps = FIELD(HOOK_MAPS);
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(maps); ++i) {
    if (PyDict_GET_SIZE(PyTuple_GET_ITEM(maps, i)) != 0) {
      MISS_FRAME();
    }
  }
  PyObject *manager = FIELD(MANAGER);
  PyObject *entry = PyObject_GetAttr(manager, n->cache_entry);
  if (entry == NULL)
    return NULL;
  ok = entry != Py_None;
  Py_DECREF(entry);
  if (!ok) {
    MISS_FRAME();
  }

  PyObject *locals = PyDict_New();
  if (locals == NULL)
    return NULL;
  PyObject *values[4] = {model};
  for (Py_ssize_t i = 0; i <= count; ++i) {
    if (i != 0)
      values[i] = args[i - 1];
    if (PyDict_SetItem(locals, PyTuple_GET_ITEM(names, i), values[i]) < 0) {
      Py_DECREF(locals);
      return NULL;
    }
  }
  /* This is Dynamo's full current guard tree, including live parameters. */
  PyObject *passed = PyObject_CallMethodOneArg(manager, n->check, locals);
  Py_DECREF(locals);
  ok = consume_truth(passed);
  if (ok < 0)
    return NULL;
  if (!ok)
    return Py_NewRef(FIELD(GUARD_MISS));
  return PyObject_Vectorcall(FIELD(REPLAY), values, count + 1, NULL);
#undef MISS_FRAME
#undef FIELD
}

static PyObject *try_frame(PyObject *self, PyObject *args) {
  if (!PyTuple_CheckExact(args)) return Py_NewRef(PyTuple_GET_ITEM(PyTuple_GET_ITEM(self, 1), MISS));
  return try_frame_values(self, ((PyTupleObject *)args)->ob_item, PyTuple_GET_SIZE(args));
}

static PyMethodDef frame_method = {"try_frame", try_frame, METH_O, NULL};

static PyObject *bind_frame(PyObject *module, PyObject *config) {
  if (!PyTuple_CheckExact(config) || PyTuple_GET_SIZE(config) != FIELD_COUNT) {
    PyErr_SetString(PyExc_TypeError, "Invalid F06 entry configuration");
    return NULL;
  }
#define FIELD(i) PyTuple_GET_ITEM(config, (i))
  PyObject *model = FIELD(MODEL), *maps = FIELD(HOOK_MAPS), *modes = FIELD(MODE_CHECKS);
  if (!PyFunction_Check(FIELD(FORWARD)) || !PyCode_Check(FIELD(CODE)) ||
      !PyDict_CheckExact(FIELD(MODEL_DICT)) || !PyTuple_CheckExact(maps) ||
      !PyTuple_CheckExact(modes) || !PyTuple_CheckExact(FIELD(NAMES)) ||
      !PyType_Check(FIELD(TENSOR_TYPE)) || !PyCallable_Check(FIELD(REPLAY))) {
    PyErr_SetString(PyExc_TypeError, "Invalid F06 frame fields");
    return NULL;
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(maps); ++i) {
    if (!PyDict_Check(PyTuple_GET_ITEM(maps, i))) {
      PyErr_SetString(PyExc_TypeError, "Expected a module hook dictionary");
      return NULL;
    }
  }
  PyTypeObject *type = Py_TYPE(model);
  if (!PyType_HasFeature(type, Py_TPFLAGS_VALID_VERSION_TAG) || type->tp_version_tag == 0)
    Py_RETURN_NONE;
  PyObject *self = Py_BuildValue("(OOOIK)", module, config, (PyObject *)type,
                                 type->tp_version_tag,
                                 (unsigned long long)((PyDictObject *)FIELD(MODEL_DICT))->ma_version_tag);
  if (self == NULL)
    return NULL;
  PyObject *method = PyCFunction_New(&frame_method, self);
  Py_DECREF(self);
  return method;
#undef FIELD
}

/* Module.__call__ still dispatches hooks, tracing, and replacement methods
 * through Torch on every miss. A hit shares exactly the guard used by forward.
 * The native cell is live across recompilation and concurrent callers. */
static PyObject *call_module(PyObject *config, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
  PyObject *wrapper = PyTuple_GET_ITEM(config, 0);
  PyObject *slow = PyTuple_GET_ITEM(config, 1);
  PyTypeObject *cls = Py_TYPE(wrapper);
  if ((PyObject *)cls != PyTuple_GET_ITEM(config, 2))
    return PyObject_Vectorcall(PyTuple_GET_ITEM(config, 4), args, nargs, kwnames);
  PyObject *version = PyTuple_GET_ITEM(config, 3);
  if (PyList_GET_SIZE(version) != 1 || !PyLong_CheckExact(PyList_GET_ITEM(version, 0)))
    return PyObject_Vectorcall(slow, args, nargs, kwnames);
  if (!PyType_HasFeature(cls, Py_TPFLAGS_VALID_VERSION_TAG) ||
      cls->tp_version_tag != PyLong_AsUnsignedLong(PyList_GET_ITEM(version, 0))) {
    /* Compiling another graph can invalidate the type tag without changing
     * dispatch. Recheck the actual descriptors once, then reuse the new tag. */
    int valid = consume_truth(PyObject_CallNoArgs(PyTuple_GET_ITEM(config, 12)));
    if (valid < 0)
      return NULL;
    if (!valid || !PyType_HasFeature(cls, Py_TPFLAGS_VALID_VERSION_TAG) || !cls->tp_version_tag)
      return PyObject_Vectorcall(PyTuple_GET_ITEM(config, 4), args, nargs, kwnames);
    PyObject *tag = PyLong_FromUnsignedLong(cls->tp_version_tag);
    if (!tag || PyList_SetItem(version, 0, tag) < 0)
      return NULL;
  }
  if (PyFunction_GET_CODE(PyMethod_GET_FUNCTION(slow)) != PyTuple_GET_ITEM(config, 13))
    return PyObject_Vectorcall(slow, args, nargs, kwnames);
  if (kwnames && PyTuple_GET_SIZE(kwnames))
    return PyObject_Vectorcall(slow, args, nargs, kwnames);

  PyObject *frame = PyCell_GET(PyTuple_GET_ITEM(config, 5));
  PyObject **dict = _PyObject_GetDictPtr(wrapper);
  if (!frame || !PyTuple_CheckExact(frame) || PyTuple_GET_SIZE(frame) != 2 ||
      !dict || !*dict || !PyDict_CheckExact(*dict))
    return PyObject_Vectorcall(slow, args, nargs, kwnames);
  PyObject *keys = PyTuple_GET_ITEM(config, 6);
  PyObject *values = PyTuple_GET_ITEM(config, 7);
  if (PyFunction_GET_CODE(PyTuple_GET_ITEM(values, 0)) != PyTuple_GET_ITEM(config, 11))
    return PyObject_Vectorcall(slow, args, nargs, kwnames);
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(keys); ++i) {
    PyObject *v = PyDict_GetItemWithError(*dict, PyTuple_GET_ITEM(keys, i));
    if (!v && PyErr_Occurred())
      return NULL;
    if (v != PyTuple_GET_ITEM(values, i) || (i && PyDict_GET_SIZE(v)))
      return PyObject_Vectorcall(slow, args, nargs, kwnames);
  }
  PyObject *attempt = Py_NewRef(PyTuple_GET_ITEM(frame, 1));
  if (!PyCFunction_Check(attempt) || PyCFunction_GET_FUNCTION(attempt) != (PyCFunction)try_frame) {
    Py_DECREF(attempt);
    return PyObject_Vectorcall(slow, args, nargs, kwnames);
  }
  PyObject *result = try_frame_values(PyCFunction_GET_SELF(attempt), args, nargs);
  Py_DECREF(attempt);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 8; i < 11; ++i) {
    if (result == PyTuple_GET_ITEM(config, i)) {
      Py_DECREF(result);
      return PyObject_Vectorcall(slow, args, nargs, kwnames);
    }
  }
  return result;
}

static PyMethodDef module_call = {"guarded_module_call", (PyCFunction)call_module, METH_FASTCALL | METH_KEYWORDS, NULL};

static PyObject *bind_dispatch(PyObject *module, PyObject *args) {
  PyObject *wrapper, *slow, *dynamic, *cell, *keys, *values, *miss, *guard_miss, *refresh, *recheck;
  if (!PyArg_ParseTuple(args, "OOOO!O!O!OOOO:bind_dispatch", &wrapper, &slow, &dynamic, &PyCell_Type, &cell,
                        &PyTuple_Type, &keys, &PyTuple_Type, &values, &miss, &guard_miss, &refresh, &recheck))
    return NULL;
  if (PyTuple_GET_SIZE(keys) != PyTuple_GET_SIZE(values) || !PyTuple_GET_SIZE(keys) ||
      !PyFunction_Check(PyTuple_GET_ITEM(values, 0)) || !PyMethod_Check(slow) ||
      !PyFunction_Check(PyMethod_GET_FUNCTION(slow)))
    Py_RETURN_NONE;
  for (Py_ssize_t i = 1; i < PyTuple_GET_SIZE(values); ++i)
    if (!PyDict_Check(PyTuple_GET_ITEM(values, i)))
      Py_RETURN_NONE;
  PyTypeObject *cls = Py_TYPE(wrapper);
  if (!PyType_HasFeature(cls, Py_TPFLAGS_VALID_VERSION_TAG) || !cls->tp_version_tag)
    Py_RETURN_NONE;
  PyObject *config = Py_BuildValue("(OOO[I]OOOOOOOOOO)", wrapper, slow, (PyObject *)cls, cls->tp_version_tag,
                                   dynamic, cell, keys, values, miss, guard_miss, refresh,
                                   PyFunction_GET_CODE(PyTuple_GET_ITEM(values, 0)), recheck,
                                   PyFunction_GET_CODE(PyMethod_GET_FUNCTION(slow)));
  if (!config)
    return NULL;
  PyObject *result = PyCFunction_New(&module_call, config);
  Py_DECREF(config);
  return result;
}

#ifdef __cplusplus
/* Shared with the optional KL3 launcher capsule. No tensor address or stream
 * is stored in the prepared packet. Each call patches its own stack copy. */
struct DenseLaunchABI {
  const char *types;
  void (*issue)(const int *, uintptr_t, uintptr_t, const int64_t *, const int64_t *);
};
struct DensePacket {
  const DenseLaunchABI *abi;
  uintptr_t function;
  int grid[6], slots[5], words;
  int64_t values[23], counts[23];
};

static void free_packet(PyObject *capsule) {
  PyMem_Free(PyCapsule_GetPointer(capsule, "f06.dense_packet.v1"));
}

static PyObject *make_packet(PyObject *config) {
  PyObject *entry = PyTuple_GET_ITEM(config, 0), *base = PyTuple_GET_ITEM(config, 1);
  if (!PyCFunction_Check(entry)) Py_RETURN_NONE;
  PyObject *owner = PyCFunction_GET_SELF(entry);
  if (!owner || !PyModule_Check(owner) || PyDict_GetItemString(PyModule_GetDict(owner), "launch") != entry) Py_RETURN_NONE;
  PyObject *capsule = PyDict_GetItemString(PyModule_GetDict(owner), "_dense_launch_abi");
  if (!capsule || !PyCapsule_IsValid(capsule, "triton.xpu.dense_launch.v1")) Py_RETURN_NONE;
  DensePacket p = {};
  p.abi = (const DenseLaunchABI *)PyCapsule_GetPointer(capsule, "triton.xpu.dense_launch.v1");
  p.words = (int)PyTuple_GET_SIZE(base) - 9;
  if (p.words < 1 || p.words > 23 || strlen(p.abi->types) != (size_t)p.words) Py_RETURN_NONE;
  for (int j = 6; j < 9; ++j)
    if (PyTuple_GET_ITEM(base, j) != Py_None) Py_RETURN_NONE;
  PyObject *meta = PyTuple_GET_ITEM(base, 5);
  int warps, ctas, shared;
  if (!PyArg_ParseTuple(meta, "iiiiii", &warps, &ctas, &shared, p.grid+3, p.grid+4, p.grid+5))
    return NULL;
  for (int i=0; i<3; ++i) {
    long n = PyLong_AsLong(PyTuple_GET_ITEM(base, i));
    if (PyErr_Occurred()) return NULL;
    if (n < 0 || n > INT_MAX) Py_RETURN_NONE;
    p.grid[i] = (int)n;
  }
  p.function = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(base, 4));
  if (PyErr_Occurred()) return NULL;
  PyObject *slots = PyTuple_GET_ITEM(config, 2), *counts = PyTuple_GET_ITEM(config, 3);
  int ptrs = 0;
  for (int j=0; j<p.words; ++j) {
    char ty = p.abi->types[j];
    PyObject *v = PyTuple_GET_ITEM(base, j+9);
    if (ty == 'p') {
      if (ptrs >= PyTuple_GET_SIZE(slots) || PyLong_AsLong(PyTuple_GET_ITEM(slots, ptrs)) != j+9
          || v != Py_None) Py_RETURN_NONE;
      p.slots[ptrs] = j;
      PyObject *count = PyTuple_GET_ITEM(counts, ptrs++);
      if (count != Py_None) {
        p.counts[j] = PyLong_AsLongLong(count);
        if (PyErr_Occurred()) return NULL;
        if (p.counts[j] < 0) Py_RETURN_NONE;
      }
    } else {
      if (!PyLong_Check(v)) Py_RETURN_NONE;
      p.values[j] = PyLong_AsLongLong(v);
      if (PyErr_Occurred()) return NULL;
      if ((ty == 'i' && (p.values[j] < INT_MIN || p.values[j] > INT_MAX))
          || (ty == 'b' && p.values[j] != 0 && p.values[j] != 1)) Py_RETURN_NONE;
    }
  }
  if (ptrs != PyTuple_GET_SIZE(slots)) Py_RETURN_NONE;
  DensePacket *owned = (DensePacket *)PyMem_Malloc(sizeof(p));
  if (!owned) return PyErr_NoMemory();
  *owned = p;
  PyObject *result = PyCapsule_New(owned, "f06.dense_packet.v1", free_packet);
  if (!result) PyMem_Free(owned);
  return result;
}
#endif

/* The KL3 launcher owns the SDK call. Keep its constant arguments here, and
 * build each call on the stack so concurrent streams never share addresses. */
static PyObject *launch_dense(PyObject *config, PyObject *const *inputs, Py_ssize_t nargs) {
  PyObject *base = PyTuple_GET_ITEM(config, 1);
  PyObject *slots = PyTuple_GET_ITEM(config, 2), *counts = PyTuple_GET_ITEM(config, 3);
  if (nargs + 1 != PyTuple_GET_SIZE(slots))
    Py_RETURN_NONE;
  PyObject *argv[32], *owned[5] = {NULL};
  PyObject *stream = NULL, *out = NULL, *result = NULL, *status = NULL;
#ifdef __cplusplus
  using DataPtr = void *(*)(PyObject *);
  const DensePacket *packet = PyTuple_GET_SIZE(config) == 13
      ? (const DensePacket *)PyCapsule_GetPointer(PyTuple_GET_ITEM(config, 12), "f06.dense_packet.v1") : NULL;
  int64_t values[23];
  if (packet) memcpy(values, packet->values, packet->words * sizeof(int64_t));
  DataPtr tensor_ptr = PyTuple_GET_SIZE(config) >= 12
      ? reinterpret_cast<DataPtr>(PyLong_AsVoidPtr(PyTuple_GET_ITEM(config, 11))) : nullptr;
#endif
  Py_ssize_t argc = PyTuple_GET_SIZE(base);
  for (Py_ssize_t i = 0; i < argc; ++i)
    argv[i] = PyTuple_GET_ITEM(base, i);
  for (Py_ssize_t i = 0; i <= nargs; ++i) {
    if (i == nargs) {
      out = PyObject_CallObject(PyTuple_GET_ITEM(config, 5), PyTuple_GET_ITEM(config, 6));
      if (!out) goto done;
    }
    PyObject *tensor = i == nargs ? out : inputs[i];
    PyObject *count = PyTuple_GET_ITEM(counts, i);
    long slot = PyLong_AsLong(PyTuple_GET_ITEM(slots, i));
    PyObject *ptr = NULL;
    uintptr_t address;
#ifdef __cplusplus
    if (tensor_ptr && ((PyObject *)Py_TYPE(tensor) == PyTuple_GET_ITEM(config, 9) ||
                       (PyObject *)Py_TYPE(tensor) == PyTuple_GET_ITEM(config, 10))) {
      // Non-affine LayerNorm repeats one live bias in three consecutive ABI slots.
      if (i > 0 && i < nargs && tensor == inputs[i - 1] && count == PyTuple_GET_ITEM(counts, i - 1)) {
        if (packet)
          values[packet->slots[i]] = values[packet->slots[i - 1]];
        else
          argv[slot] = owned[i] = Py_NewRef(owned[i - 1]);
        continue;
      }
      try {
        // Inductor's accessor reads the live TensorImpl, including its current storage offset.
        void *raw = tensor_ptr(tensor);
        address = reinterpret_cast<uintptr_t>(raw);
        if (packet) {
          if ((address & 15) || (!address && packet->counts[packet->slots[i]])) {
            result = Py_NewRef(Py_None);
            goto done;
          }
          values[packet->slots[i]] = (int64_t)address;
          continue;
        }
        ptr = PyLong_FromVoidPtr(raw);
        if (!ptr || PyErr_Occurred()) { Py_XDECREF(ptr); goto done; }
      } catch (...) {
        // The ordinary tensor method translates Torch exceptions into Python errors.
        if (PyErr_Occurred()) goto done;
      }
    }
#endif
    if (!ptr) {
      ptr = PyObject_CallOneArg(PyTuple_GET_ITEM(config, 4), tensor);
      if (!ptr) goto done;
      address = (uintptr_t)PyLong_AsUnsignedLongLong(ptr);
      if (PyErr_Occurred()) { Py_DECREF(ptr); goto done; }
    }
#ifdef __cplusplus
    if (packet) {
      Py_DECREF(ptr);
      if ((address & 15) || (!address && packet->counts[packet->slots[i]])) {
        result = Py_NewRef(Py_None);
        goto done;
      }
      values[packet->slots[i]] = (int64_t)address;
      continue;
    }
#endif
    owned[i] = ptr;
    if (address & 15) {
      result = Py_NewRef(Py_None);
      goto done;
    }
    if (count != Py_None) {
      owned[i] = PyTuple_Pack(2, ptr, count);
      Py_DECREF(ptr);
      if (!owned[i]) goto done;
    }
    argv[slot] = owned[i];
  }
  stream = PyObject_CallOneArg(PyTuple_GET_ITEM(config, 7), PyTuple_GET_ITEM(config, 8));
  if (!stream) goto done;
  argv[3] = stream;
#ifdef __cplusplus
  if (packet) {
    uintptr_t handle = (uintptr_t)PyLong_AsUnsignedLongLong(stream);
    if (PyErr_Occurred()) goto done;
    Py_BEGIN_ALLOW_THREADS
    packet->abi->issue(packet->grid, handle, packet->function, values, packet->counts);
    Py_END_ALLOW_THREADS
    if (!PyErr_Occurred()) {result = out; out = NULL;}
    goto done;
  }
#endif
  status = PyObject_Vectorcall(PyTuple_GET_ITEM(config, 0), argv, argc, NULL);
  if (status) {
    Py_DECREF(status);
    result = out;
    out = NULL;
  }
done:
  for (Py_ssize_t i = 0; i <= nargs; ++i) Py_XDECREF(owned[i]);
  Py_XDECREF(stream);
  Py_XDECREF(out);
  return result;
}

static PyMethodDef dense_call = {"launch_dense", (PyCFunction)launch_dense, METH_FASTCALL, NULL};

static PyObject *bind_dense(PyObject *module, PyObject *config) {
  if (!PyTuple_CheckExact(config) || PyTuple_GET_SIZE(config) != 9) Py_RETURN_NONE;
  PyObject *base = PyTuple_GET_ITEM(config, 1), *slots = PyTuple_GET_ITEM(config, 2);
  PyObject *counts = PyTuple_GET_ITEM(config, 3);
  if (!PyTuple_CheckExact(base) || PyTuple_GET_SIZE(base) < 9 || PyTuple_GET_SIZE(base) > 32 ||
      !PyTuple_CheckExact(slots) || PyTuple_GET_SIZE(slots) < 2 || PyTuple_GET_SIZE(slots) > 5 ||
      !PyTuple_CheckExact(counts) || PyTuple_GET_SIZE(counts) != PyTuple_GET_SIZE(slots) ||
      !PyTuple_CheckExact(PyTuple_GET_ITEM(config, 6))) Py_RETURN_NONE;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(slots); ++i) {
    PyObject *v = PyTuple_GET_ITEM(slots, i);
    if (!PyLong_CheckExact(v)) Py_RETURN_NONE;
    long slot = PyLong_AsLong(v);
    if (PyErr_Occurred()) return NULL;
    if (slot < 9 || slot >= PyTuple_GET_SIZE(base) || PyTuple_GET_ITEM(base, slot) != Py_None)
      Py_RETURN_NONE;
  }
  return PyCFunction_New(&dense_call, config);
}

#ifdef __cplusplus
static PyObject *bind_native_dense(PyObject *module, PyObject *args) {
  PyObject *config, *tensor_type, *parameter_type;
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyTuple_Type, &config, &PyType_Type, &tensor_type,
                        &PyType_Type, &parameter_type)) return NULL;
  PyObject *fallback = bind_dense(module, config);
  if (!fallback || fallback == Py_None) return fallback;
  PyObject *guards = PyImport_ImportModule("torch._C._dynamo.guards");
  if (!guards) {
    if (PyErr_ExceptionMatches(PyExc_ImportError)) { PyErr_Clear(); return fallback; }
    Py_DECREF(fallback);
    return NULL;
  }
  PyObject *addr = PyObject_GetAttrString(guards, "_torchinductor_pyobject_tensor_data_ptr");
  Py_DECREF(guards);
  if (!addr && PyErr_ExceptionMatches(PyExc_AttributeError)) { PyErr_Clear(); return fallback; }
  if (!addr) { Py_DECREF(fallback); return NULL; }
  if (!PyLong_CheckExact(addr) || !PyLong_AsVoidPtr(addr)) {
    Py_DECREF(addr);
    if (PyErr_Occurred()) { Py_DECREF(fallback); return NULL; }
    return fallback;
  }
  Py_DECREF(fallback);
  PyObject *packet = make_packet(config);
  if (!packet) { Py_DECREF(addr); return NULL; }
  PyObject *extra = packet == Py_None ? PyTuple_Pack(3, tensor_type, parameter_type, addr)
                                      : PyTuple_Pack(4, tensor_type, parameter_type, addr, packet);
  Py_DECREF(packet);
  Py_DECREF(addr);
  if (!extra) return NULL;
  PyObject *bound = PySequence_Concat(config, extra);
  Py_DECREF(extra);
  if (!bound) return NULL;
  PyObject *method = PyCFunction_New(&dense_call, bound);
  Py_DECREF(bound);
  return method;
}
#endif

/* Positional call and result[0], for a fully matched Dynamo replay program. */
static PyObject *single_output(PyObject *config, PyObject *const *values, Py_ssize_t nargs) {
  Py_ssize_t count = PyLong_AsSsize_t(PyTuple_GET_ITEM(config, 1));
  if (nargs != count + 1) {
    PyErr_SetString(PyExc_TypeError, "Unexpected number of guarded frame arguments");
    return NULL;
  }
  PyObject *result = PyObject_Vectorcall(PyTuple_GET_ITEM(config, 0), values + 1, count, NULL);
  if (!result) return NULL;
  PyObject *out = PyObject_GetItem(result, PyTuple_GET_ITEM(config, 2));
  Py_DECREF(result);
  return out;
}

static PyMethodDef single_call = {"single_output", (PyCFunction)single_output, METH_FASTCALL, NULL};

static PyObject *bind_single_output(PyObject *module, PyObject *args) {
  PyObject *graph;
  int nargs;
  if (!PyArg_ParseTuple(args, "Oi", &graph, &nargs)) return NULL;
  if (!PyCallable_Check(graph) || (nargs != 2 && nargs != 3)) Py_RETURN_NONE;
  PyObject *config = Py_BuildValue("(Oii)", graph, nargs, 0);
  if (!config) return NULL;
  PyObject *bound = PyCFunction_New(&single_call, config);
  Py_DECREF(config);
  return bound;
}

static PyMethodDef methods[] = {{"bind_single_output", bind_single_output, METH_VARARGS, NULL},
                                {"bind", bind_frame, METH_O, NULL},
                                {"bind_dense", bind_dense, METH_O, NULL},
#ifdef __cplusplus
                                {"bind_native_dense", bind_native_dense, METH_VARARGS, NULL},
#endif
                                {"bind_dispatch", bind_dispatch, METH_VARARGS, NULL}, {NULL}};
static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT, "_p800_frame_guard", NULL, sizeof(Names), methods,
  NULL, traverse, clear_names, NULL
};

PyMODINIT_FUNC PyInit__p800_frame_guard(void) {
  PyObject *module = PyModule_Create(&module_def);
  if (module == NULL)
    return NULL;
  Names *n = (Names *)PyModule_GetState(module);
#define INTERN(member, name) if (!(n->member = PyUnicode_InternFromString(name))) goto fail
  INTERN(stance, "_stance");
  INTERN(stance_name, "stance");
  INTERN(backend, "backend");
  INTERN(skip_guards, "skip_guard_eval_unsafe");
  INTERN(default_stance, "default");
  INTERN(profiler_flag, "_is_profiler_enabled");
  INTERN(global_hooks, "_has_any_global_hook");
  INTERN(cache_entry, "cache_entry");
  INTERN(check, "check");
#undef INTERN
  return module;
fail:
  Py_DECREF(module);
  return NULL;
}
