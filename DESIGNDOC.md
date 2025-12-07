# Design: Two-Stage Product Quantization for HNSW in pgvector v0.8.0

This document describes a patch for pgvector v0.8.0 that adds a two-stage Product Quantization (PQ) design to HNSW to achieve
(1) traversal-time pruning using compact neighbor metadata and (2) element payload compression via residual PQ with rotation-based optimization.
The goal is to reduce random I/O and distance computations while controlling index size growth and minimizing reconstruction error.
This design is based on a vector encoding strategy proposed in [1].

## Summary

- **Two-Stage Product Quantization**
  - **Stage 1 (Neighbor PQ)**: Each vector is PQ-encoded [2] into a 16-byte code (16 subspaces, 1-byte per subspace). The code is stored alongside each neighbor entry (neighbor metadata) and used at query time for distance estimation to prune candidates
  - **Stage 2 (Element PQ)**: The residual of the input vector with respect to the Stage 1 reconstruction is PQ-encoded and stored in the element tuple. To minimize quantization error, a learned rotation matrix $R$ is applied before PQ encoding [3] to decorrelate vector dimensions (this rotation-based approach is known as Optimized Product Quantization or OPQ). This preserves recall while keeping the overall index size modest despite the added neighbor metadata. Note that approximating an original vector requires adding the Stage-2 residual vector (decoded from the element tuple) to the Stage-1 reconstruction decoded from the 16-byte PQ code stored in the neighbor entry
- **Training and Persistence**: Codebooks for Neighbor PQ and rotation matrix + codebooks for Element PQ are pre-trained at index build time using samples drawn from the indexed table (similar to the IVFFlat implementation in pgvector) and persisted in dedicated codebook pages; the metapage is extended with PQ layout and bootstrap metadata

## Motivation

In PostgreSQL, data structures are managed in fixed-size blocks (e.g., 8 KiB), and HNSW’s vertices and edges are mapped to these blocks.
During greedy search in HNSW, adjacent vertices often reside on different blocks, causing frequent random block accesses.
In an RDBMS, this not only increases I/O but also exacerbates lock/latch contention on shared buffers and pages under concurrent query workloads.
To mitigate these issues, prior work [4] has confirmed that candidate pruning in pgvector reduces random block accesses
by attaching compact metadata to neighbors and ranking them by estimated distance so that fewer blocks need to be accessed.
However, a salient trade-off of this approach, observed in prior work, is the storage overhead introduced by per-neighbor metadata.
On SIFT1M, the vanilla pgvector index occupies 781 MiB, whereas enabling the 16-byte neighbor metadata inflates the index to 1313 MiB,
corresponding to an increase of approximately 68%. Therefore, this design aims to preserve the benefits of candidate pruning
while curbing index-size growth arising from per-neighbor metadata.

## Scope and Non-Goals

- Scope
  - HNSW in pgvector v0.8.0 only
    - Implementation note: place new logic in new files and keep edits to existing code minimal to ease future upgrades and rebases to newer pgvector versions
  - Two-stage PQ: Stage-1 for traversal-time pruning (candidate pruning with neighbor metadata) and Stage-2 for element payload compression (to control index size growth from per-neighbor metadata)
    - Stage-1 PQ codebooks and Stage-2 rotation matrix + PQ codebooks are trained at build time from table samples, shared in parallel build, and stored in dedicated pages just after the meta page
    - Stage-2 Element PQ uses iterative optimization (rotation matrix R and PQ codebooks) to minimize reconstruction error by decorrelating dimensions before quantization
  - When Two-Stage PQ is disabled, preserve legacy pgvector behavior (no PQ-based pruning/metadata and no element compression)
- Non-Goals
  - Online/incremental codebook training or adaptive updates at query time
  - On-disk compatibility with existing HNSW indexes (rebuild required)
  - Exact recall equivalence guarantees (approximation may slightly reduce recall)
  - Modifications to PostgreSQL locking/buffer/WAL/replication subsystems

## User-Facing Options

### Index Creation Options (CREATE INDEX)
  - `neighbor_metadata` (bool, default on): Store 16-byte PQ codes per neighbor to enable candidate pruning
  - `enable_compression` (bool, default on): Store element payloads as PQ codes (residual PQ when neighbor_metadata=on; direct PQ when neighbor_metadata=off)

### Session Parameters (SET)
- Build-time parameters
  - `hnsw.opq_rotation_iters` (int, default 10, range 1-100): Number of rotation optimization iterations for Stage-2 Element PQ
- Query-time parameters
  - `hnsw.candidate_pruning` (bool, default on): Enable candidate pruning with estimated distances
  - `hnsw.distance_computation_topk` (int, default 3, upper bound 2*m): For a neighbor list, compute precise distances for only the top- $k$ by estimated distance

## Index Build, Training, and Persistence

- Sampling and Training: During index build, a sample of vectors is collected for training
  - Stage 1 (Neighbor PQ): Divides each vector into 16 subspaces and trains 256 centroids per subspace via k-means, yielding 16-byte neighbor metadata (1-byte per subspace). This stage is only performed when neighbor_metadata=on
  - Stage 2 (Element PQ): Uses rotation-based optimization to encode vectors. Let $d$ denote the vector dimensions. Divides each vector into $M$ subspaces where each subspace has 8 dimensions (thus $M = \lceil d / 8 \rceil$), and trains 256 centroids per subspace. Let $X$ denote the input vectors for training:
    - **Residual PQ (`neighbor_metadata=on`)**: $X$ = residual vectors (difference between original vectors and Stage-1 reconstructions)
    - **Direct PQ (`neighbor_metadata=off`)**: $X$ = original sample vectors directly
    - Iterative optimization alternates between:
      1. Rotating input vectors by current matrix $R$: $X_{\text{rot}} = X \cdot R^T$ (for Residual PQ, compute residuals first in each iteration)
      2. Training PQ codebooks on $X_{\text{rot}}$ via k-means (256 centroids per subspace, yielding $M$-byte codes with 1-byte per subspace)
      3. Encoding/decoding $X_{\text{rot}}$ to get reconstruction $Y$
      4. Updating $R$ via Procrustes analysis (SVD-based): $R_{\text{new}} = U \cdot V^T$ where $\text{SVD}(Y^T \cdot X) = U \cdot \Sigma \cdot V^T$
  - Number of rotation iterations controlled by `hnsw.opq_rotation_iters` (default: 10)
  - Matrix operations use BLAS and SVD uses LAPACK (`-llapacke -llapack -lblas`)
  - Implementation note: Reuse the sampling and k-means logic from pgvector's IVFFlat to train subspace codebooks
- Persistence: Trained codebooks for neighbor metadata, rotation matrix, and PQ codebooks for element residuals are written as dedicated "codebook pages" preceded by HNSW graph data
- Parallel Build: PQ state (codebooks, rotation matrix, and layout) is shared with workers and restored from shared memory

## On-Disk Structures

- Metapage
  - The metapage in the first block additionally stores:
    - Codebook info ($M$, sub-dimensions, the start block number of codebook pages, and #pages)
    - Rotation matrix info (offset and size for rotation matrix $R$: $d \times d$ floats)
    - Entry metadata (16-byte neighbor metadata for the entry point) used when residual PQ is enabled (`neighbor_metadata=on` and `enable_compression=on`)
    - The start block number of HNSW graph data
- Centroid tuples
  - New tuple type (HNSW_CENTROID_TUPLE_TYPE) to store subspace centroids laid out consecutively in codebook pages
  - For Element PQ, stores both rotation matrix $R$ and PQ centroids in rotated space
- Neighbor tuples
  - Layout changed from a plain array of ItemPointerData to entries[] where each entry holds ItemPointer + 16-byte neighbor metadata when neighbor metadata enabled (`neighbor_metadata=on`)
  - Neighbor metadata uses standard PQ (without rotation) for simplicity and minimal overhead
- Element tuples
  - Layout changed from original Vector datum (no compression; varlena) to data[] for $M$-byte PQ code with rotation:
    - Residual PQ: when `neighbor_metadata=on`, encodes residual from Stage-1 reconstruction
    - Direct PQ: when `neighbor_metadata=off`, encodes original vector directly
  - Storage format: uint8 codes[ $M$ ] (1 byte per subspace)

## Search-Time Algorithm

- If candidate pruning is enabled (`hnsw.candidate_pruning=on`) and neighbor metadata is present (`neighbor_metadata=on`):
  - Read 16-byte PQ codes of unvisited neighbors and compute estimated distances using its codebook (Stage-1 PQ)
  - Evaluate top- $k$ (`hnsw.distance_computation_topk`) candidates:
    1. Decode Stage-1 PQ code from neighbor metadata
    2. Decode Stage-2 PQ code from element tuple: decode PQ centroids then apply inverse rotation ($R^T$)
    3. Add Stage-1 and Stage-2 reconstructions to get final approximation
    4. Compute distance using the approximated vector and push to candidate queue (NOTE: original vector is lossy compressed, so exact distance cannot be computed here)
  - Postpone remaining candidates to estimated-candidate queue
  - After HNSW greedy search finished, check the estimated-candidate queue and evaluate distances for any remaining unvisited candidates before finalizing results

## Insertion, Update, and Vacuum

- Insert
  - Compute Stage-1 neighbor code (16-byte PQ) when `neighbor_metadata=on`
  - Compute Stage-2 element code:
    - Residual PQ (`neighbor_metadata=on`): compute residual from Stage-1 reconstruction, rotate by $R$, then PQ-encode
    - Direct PQ (`neighbor_metadata=off`): rotate original vector by $R$, then PQ-encode
  - Write element tuple and neighbor tuples accordingly
  - Update entryMetadata in the metapage if the entry point replaced
- Update/Graph maintenance
  - Neighbor tuple writes include the 16-byte metadata when enabled (`neighbor_metadata=on`)
  - Element tuple encoding/decoding applies rotation
- Vacuum
  - When `enable_compression=on`, element tuples store only PQ codes (residual PQ when `neighbor_metadata=on`; direct PQ when `neighbor_metadata=off`) instead of the original vector; because per-element Stage‑1 codes (stored as neighbor metadata) are not stored with the element (only the entry point's code lives in the metapage) and lossy PQ reconstruction would bias neighbor reselection and accumulate error, vacuum reads the original vector from the heap to recompute distances and repair neighbor lists accurately

## Compatibility

- On-disk layout has changed (metapage, neighbor tuples, codebook pages). Existing HNSW indexes are not binary compatible and must be rebuilt

## Performance Expectations and Trade-offs

- Expected benefits
  - Fewer random I/Os and distance evaluations due to traversal-time pruning using Stage-1 codes stored as neighbor metadata
- Trade-offs
  - Build-time overhead: codebook training via k-means for both stages, and additional rotation optimization with SVD for Stage-2 (samples + rotation matrix + codebooks in memory)
  - Search-time overhead: encoding/decoding requires matrix-vector multiplication (rotation) in addition to PQ operations
  - Approximation can reduce recall slightly; tuning `hnsw.distance_computation_topk` mitigates this

## References

 - [1] M. Douze, A. Sablayrolles and H. Jégou, "Link and Code: Fast Indexing with Graphs and Compact Regression Codes," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 3646-3654, doi: 10.1109/CVPR.2018.00384.
 - [2] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128.
 - [3] T. Ge, K. He, Q. Ke and J. Sun, "Optimized Product Quantization for Approximate Nearest Neighbor Search," 2013 IEEE Conference on Computer Vision and Pattern Recognition, Portland, OR, USA, 2013, pp. 2946-2953, doi: 10.1109/CVPR.2013.379.
 - [4] A patch to improve the pgvector implementation of HNSW indices with candidate pruning for minimizing disk block accesses, https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch.