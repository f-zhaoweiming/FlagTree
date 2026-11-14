Compiler-generated XPU resource partitions (experimental)
========================================================

The opt-in ``graph_fusion.resource_backend`` is a ``torch.compile`` backend for
previously measured and accepted inference partitions. Native Inductor generates
all device kernels. The partitioner copies existing FX operations and bindings;
it does not supply arithmetic kernels, change precision, or manually tune
MatMul tile/warp configurations.

Resource contract
-----------------

The actual XPU driver identifies Baidu targets. A CUDA-compatible device name
alone is insufficient. Each generated kernel has one requested resource:
``cluster`` or ``sdnn``. An SDNN-required operation conflicts with a Cluster
request, and a TLE Cluster kernel conflicts with an SDNN request. The backend
rejects these conflicts. Requests participate in the compiler cache identity
and are checked against compiled binary metadata. Existing ``auto`` behavior
remains the default.

A graph may contain several kernels using different resources. Candidate
regions are contiguous intervals of compute nodes and may span multiple
operations. Multi-output region boundaries preserve residual dependencies.
The planner compares measured region costs, including output materialization,
and never infers costs for unmeasured or invalid regions.

MatMul paths
------------

* ``external`` retains native library MatMul and lets Inductor generate the
  postprocessing kernels. The library call remains an unverified resource
  boundary; an operator name is not evidence of SDNN execution.
* ``template`` requests Inductor's existing Triton MatMul templates and native
  candidate configurations. Compiler support and numerical validation still
  apply. Failure to generate a template is not a successful fused MatMul.

Compilation and native autotuning must finish before device graph capture.
Generated function bodies remain unchanged by resource annotation.

Acceptance and guards
---------------------

Profiles are scoped to exact graph/input metadata and an explicit environment
fingerprint. Compile, ordinary-value, repeatability, special-value, immutability
and compiler-resource evidence checks must pass. External calls are excluded
from verified-resource observations. A complete selected plan additionally
requires end-to-end correctness and performance acceptance. Its identity must
be explicitly supplied in ``accepted_plan_ids`` before deployment::

    backend = resource_backend(
        observations,
        environment_key=environment_key,
        accepted_plan_ids=accepted_plan_ids,
        matmul_path="template",
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

Native Dynamo guards remain around the prepared callable. Unrecognized devices,
unsupported graphs, missing profiles or unaccepted plans use native whole-graph
Inductor. The prepared wrapper performs data bindings and compiled region calls;
it does not interpret or repartition the graph on each invocation.

Current validation status
-------------------------

CPU contract tests and the isolated P800 installation's host API checks have
passed. Device compilation, special-value coverage and resource/performance
conclusions remain pending a healthy idle P800. No default resource preference
or validated performance profile ships with this experimental interface.
