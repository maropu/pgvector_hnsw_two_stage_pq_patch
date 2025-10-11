![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Build and test](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/BuildAndTests.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/BuildAndTests.yml)
[![Prebuilt binaries](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/ExtensionDistribution.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch/actions/workflows/ExtensionDistribution.yml)

## What this patch does and how to apply it?

This patch adds a two-stage Product Quantization (PQ) logic to the pgvector HNSW implementation.
Stage 1 stores a 16-byte PQ code per neighbor (neighbor metadata) to estimate distances during traversal and prune candidates.
Stage 2 compresses element payloads using residual PQ to offset the neighbor-metadata overhead and prevent index-size bloat while preserving recall.
PQ codebooks are trained at build time by reusing the IVFFlat’s sampling and k-means logic in pgvector, persisted in dedicated codebook pages, and referenced via metapage extensions.
See [DESIGNDOC.md](./DESIGNDOC.md) for the detailed implementation design.

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

## TODO

 - Update [DESIGNDOC.md](./DESIGNDOC.md) and proceed with implementation accordingly
 - Add experimental results to README.md
