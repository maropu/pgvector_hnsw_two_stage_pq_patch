![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Build and test](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/BuildAndTests.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/BuildAndTests.yml)
[![Prebuilt binaries](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/ExtensionDistribution.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/ExtensionDistribution.yml)

## What this patch does and how to apply it?

XXX

Apply the patch to pgvector and compile it as described below:

```shell
// Cehckout pgvector v0.8.0
$ git clone --depth 1 https://github.com/pgvector/pgvector.git
$ cd pgvector
$ git fetch --tags --depth 1 origin "v0.8.0"
$ git checkout "v0.8.0"

// Compile and install pgvector w/the patch
$ patch -p1 < pgvector_v0.8.0_hnsw_two_stage_pq.patch
$ make
$ make install
```

Note that **this patch is incompatible with the pgvector’s original index data format** because it adds 16 bytes per-neighbor metadata, and
it currently supports only the L2 distance (vector_l2_ops) on single-precision floating-point vectors.

### Additional options

#### Index options

Specify HNSW additional two index parameters:

- `neighbor_metadata` - whether to store neighbor metadata to estimate distances (on by default)
- `enable_compression` - whether to encode element vectors with PQ (on by default)

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64, neighbor_metadata = on, enable_compresssion = on);
```

#### Query options

Specify HNSW additional two query parameters:

- `hnsw.candidate_pruning` - enables candidate pruning for faster scans (on by default)
- `hnsw.distance_computation_topk ` - sets the number of neighbors to compute precise distances when using distance estimation (3 by default)

```sql
SET hnsw.distance_computation_topk = 3;
```

A higher value provides better recall at the cost of block accesses.

## Benchmark results

XXX

## TODO

 - XXX


## References

 - [1] XXX

