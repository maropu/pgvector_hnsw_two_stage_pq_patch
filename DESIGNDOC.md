# Design: Two-Stage Product Quantization for HNSW in pgvector v0.8.0

This document describes a patch for pgvector v0.8.0 that adds a two-stage Product Quantization (PQ) design to HNSW to achieve
(1) traversal-time pruning using compact neighbor metadata and (2) element payload compression via residual PQ.
The goal is to reduce random I/O and distance computations while controlling index size growth.
This design is based on a vector encoding strategy proposed in [1].

## Summary

- Stage 1 (Neighbor PQ): Each vector is PQ-encoded into a 16-byte code (M subspaces, 1-byte per subspace). The code is stored alongside each neighbor entry (neighbor metadata) and used at query time for distance estimation to prune candidates.
- Stage 2 (Element PQ): The residual of the input vector with respect to the Stage 1 reconstruction is PQ-encoded and stored in the element tuple. This preserves recall while keeping the overall index size modest despite the added neighbor metadata. Note that approximating an original vector requires adding the Stage-2 residual vector (decoded from the element tuple) to the Stage-1 reconstruction decoded from the 16-byte PQ code stored in the neighbor entry.
- Codebooks for both Neighbor PQ and Element PQ are pre-trained at index build time using samples drawn from the indexed table (similar to the IVFFlat implementation in pgvector) and persisted in dedicated codebook pages; the metapage is extended with PQ layout and bootstrap metadata.

## Motivation

In PostgreSQL, data structures are managed in fixed-size blocks (e.g., 8 KiB), and HNSW’s vertices and edges are mapped to these blocks.
During greedy search in HNSW, adjacent vertices often reside on different blocks, causing frequent random block accesses.
In an RDBMS, this not only increases I/O but also exacerbates lock/latch contention on shared buffers and pages under concurrent query workloads.
To mitigate these issues, prior work [2] has confirmed that candidate pruning in pgvector reduces random block accesses
by attaching lightweight metadata to neighbors and ranking them by estimated distance so that fewer blocks need to be accessed.
However, a salient trade-off of this approach, observed in prior work, is the storage overhead introduced by per-neighbor metadata.
On SIFT1M, the vanilla pgvector index occupies 781 MiB, whereas enabling the 16-byte neighbor metadata inflates the index to 1313 MiB,
corresponding to an increase of approximately 68%. Therefore, this design aims to preserve the benefits of candidate pruning
while curbing index-size growth arising from per-neighbor metadata.

## Scope and Non-Goals

- Scope
  - HNSW in pgvector v0.8.0 only
    - Implementation note: place new logic in new files and keep edits to existing code minimal to ease future upgrades and rebases to newer pgvector versions
  - Two-stage PQ: Stage-1 for traversal-time pruning and Stage-2 for element payload compression
  - PQ Codebooks are trained at build time from table samples, shared in parallel build, and stored in dedicated pages just after the meta page
  - When Two-Stage PQ is disabled, preserve legacy pgvector behavior (no PQ-based pruning/metadata and no element compression)
- Non-Goals
  - Online/incremental codebook training or adaptive updates at query time
  - On-disk compatibility with existing HNSW indexes (rebuild required)
  - Exact recall equivalence guarantees (approximation may slightly reduce recall)
  - Modifications to PostgreSQL locking/buffer/WAL/replication subsystems

## User-Facing Options

- Index options
  - neighbor_metadata (bool, default on): Store 16-byte PQ codes per neighbor to enable candidate pruning
  - enable_compression (bool, default on): Store element payloads as PQ codes (residual PQ when neighbor metadata is on; direct PQ otherwise)
- Query options
  - hnsw.candidate_pruning (bool, default on): Enable candidate pruning with estimated distances
  - hnsw.distance_computation_topk (int, default 3, upper bound 2*m): For a neighbor list, compute precise distances for only the top-k by estimated distance

## Index Build, Training, and Persistence

- Sampling and Training: During index build, a sample of vectors is collected and k-means produces K centroids for each subspace (typically K=256, 1-byte per subspace)
  - Implementation note: Reuse the sampling and k-means logic from pgvector's IVFFlat to train subspace codebooks
- Persistence: Trained codebooks for both neighbor metadata and element residuals are written as dedicated “codebook pages” preceded by HNSW graph data
- Parallel Build: PQ state (codebooks and layout) is shared with workers and restored from shared memory

## On-Disk Structures

- Metapage
  - The metapage in the first block additionally stores:
    - Codebook info (M, sub-dimensions, the start block number of codebook pages, and #pages)
    - Entry metadata (16-byte neighbor metadata for the entry point) used when residulal PQ is enabled (neighbor_metadata=on and enable_compression=on)
    - The start block number of HNSW graph data
- Centroid tuples
  - New tuple type (HNSW_CENTROID_TUPLE_TYPE) to store subspace centroids laid out consecutively in codebook pages
- Neighbor tuples
  - Layout changed from a plain array of ItemPointerData to entries[] where each entry holds ItemPointer + 16-byte neighbor metadata when neighbor metadata enabled (neighbor_metadata=on)
- Element tuples
  - Layout changed from original Vector datum (no compression; varlena) to data[] for M-byte PQ code (direct PQ) or M-byte residual PQ code

## Search-Time Algorithm

- If candidate pruning is enabled (hnsw.candidate_pruning=on) and neighbor metadata is present (neighbor_metadata=on):
  - Read 16-byte PQ codes of unvisited neighbors and compute estimated distances using its codebook
  - Evaluate top-k (hnsw.distance_computation_topk) candidates reconstructed from 16-byte PQ codes and residual PQ codes (two-stage PQ decoding) and puth them into a candiate queue; postpone the rest by pushing them into an estimated-candidate queue
  - After HNSW greedy search finished, check the estimated-candidate queue and evaluate distances for any remaining unvisited candidates before finalizing results

## Insertion, Update, and Vacuum

- Insert
  - Compute Stage-1 neighbor code (16-byte) and Stage-2 residual PQ (or direct PQ if neighbor_metadata=off). Write element tuple and neighbor tuples accordingly.
  - Update entryMetadata in the metapage if the entrhy point replaced
- Update/Graph maintenance
  - Neighbor tuple writes include the 16-byte metadata when enabled (neighbor_metadata=on)
- Vacuum
  - When enable_compression=on, element tuples store only PQ codes (residual PQ when neighbor_metadata=on, direct PQ otherwise) instead of the original vector; because per-element Stage‑1 codes (stored as neighbor metadata) are not stored with the element (only the entry point’s code lives in the metapage) and lossy PQ reconstruction would bias neighbor reselection and accumulate error, vacuum reads the original vector from the heap to recompute distances and repair neighbor lists accurately

## Compatibility

- On-disk layout has changed (metapage, neighbor tuples, codebook pages). Existing HNSW indexes are not binary compatible and must be rebuilt.

## Performance Expectations and Trade-offs

- Expected benefits
  - Fewer random I/Os and distance evaluations due to traversal-time pruning using Stage-1 codes stored as neighbor metadata
  - Smaller element payloads via residual PQ offset the 16-byte neighbor metadata overhead and prevent index-size bloat
- Trade-offs
  - Approximation can reduce recall slightly; tuning hnsw.distance_computation_topk mitigates this
  - Training adds build-time cost and some memory usage

## References

 - [1] M. Douze, A. Sablayrolles and H. Jégou, "Link and Code: Fast Indexing with Graphs and Compact Regression Codes," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 3646-3654, doi: 10.1109/CVPR.2018.00384.
 - [2] A patch to improve the pgvector implementation of HNSW indices with candidate pruning for minimizing disk block accesses, https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch.