# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move non-Parameter tensors out of GMS storage before weight publication.

Dynamo Snapshot invokes this copy while a whole engine is paused, before GMS
publishes and unmaps its weights. Restore preserves the same Python/Torch process
state, TensorImpls, post-partition StorageImpl graph, layouts, allocation IDs,
and CUDA VA reservations. Model construction, loading, and this copy do not run
again after restore. The rank-local GMS sidecar survives separately and retains
committed weight backing. Copies made with Torch's default allocator are
ordinary process-owned Snapshot state. KV physical backing and contents are not
retained; fresh backing is mapped at preserved VAs.

One GMS source storage may contain Parameters and other tensor views::

    before
    GMS storage: [0-------------63]
                  |Parameter|
                       |view A------|
                            |view B|
                                          |view C|

    after
    GMS RO:      [0-------------63]
                  |Parameter|
    default #1:       [view A------]
                          [view B]       (overlap and relative offsets preserved)
    default #2:                            [view C]

Mixed Parameter/non-Parameter aliasing is deliberately severed. Each overlapping
connected component of non-empty bounding storage byte ranges gets one compact
copy; disjoint ranges get separate storage. A copied component may rebase its
absolute storage offset, but relative aliases and offsets within it are preserved.
"""


from gpu_memory_service.client.torch.storage import (  # noqa: F401 -- compatibility exports
    _clone_storage_spans_and_rebind_tensors,
    copy_non_parameter_tensors_to_default_allocator,
    tensor_storage_byte_bounds,
)
