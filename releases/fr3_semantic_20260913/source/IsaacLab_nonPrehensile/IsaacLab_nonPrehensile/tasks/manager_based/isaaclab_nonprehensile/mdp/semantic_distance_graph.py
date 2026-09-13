"""Isolated fixed-shape CUDA graph; exact operations, fresh output ownership."""
import torch


class SemanticDistanceGraph:
    def __init__(self, function, left, right, semantic_masks):
        tensors = (left, right, *semantic_masks)
        if left.device.type != 'cuda' or left.shape[0] != 1 or any(x.requires_grad for x in tensors):
            raise ValueError('Graph is restricted to single-environment CUDA inference')
        if any(x.device != left.device for x in tensors):
            raise ValueError('Mixed devices')
        if any(not x.is_contiguous() for x in tensors):
            raise ValueError('Capture requires contiguous inputs; use eager evaluation for other layouts')
        self.signature = [(x.shape, x.dtype, x.device) for x in tensors]
        self.static = tuple(x.detach().clone() for x in tensors)
        self.function = function
        with torch.cuda.device(left.device):
            stream = torch.cuda.Stream(device=left.device)
            stream.wait_stream(torch.cuda.current_stream(left.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(self.static[0], self.static[1], semantic_masks=self.static[2:])
            torch.cuda.current_stream(left.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = function(self.static[0], self.static[1], semantic_masks=self.static[2:])

    def __call__(self, left, right, *, semantic_masks):
        tensors = (left, right, *semantic_masks)
        if [(x.shape, x.dtype, x.device) for x in tensors] != self.signature or any(x.requires_grad for x in tensors):
            raise ValueError('Changed graph shape, dtype, device, or gradient contract')
        # cdist may choose different arithmetic for other memory layouts.
        # Preserve the caller's original operation path in that case.
        if any(not x.is_contiguous() for x in tensors):
            return self.function(left, right, semantic_masks=semantic_masks)
        for dst, src in zip(self.static, tensors):
            dst.copy_(src)
        self.graph.replay()
        # A later call must never mutate a geometry/contact cache from this call.
        distances, indices, minima = self.outputs
        return distances.clone(), indices.clone(), tuple(x.clone() for x in minima)
