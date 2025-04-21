import numpy as np
import torch
from reedsolo import RSCodec
from typing import List, Tuple, Any

def encode_shard(tensor: torch.Tensor, n_fragments: int, nsym: int) -> List[bytes]:
    """
    Encode a tensor shard into fragments using Reed–Solomon coding.

    Args:
        tensor (torch.Tensor): The tensor (parameter or gradient shard) to encode.
        n_fragments (int): Total number of fragments to split the encoded data into.
        nsym (int): Number of parity symbols (redundancy); higher nsym means more fault tolerance.

    Returns:
        List[bytes]: A list of encoded fragments, each as a bytes object.
    """
    # Convert the tensor to a NumPy array with type float32 and then to bytes.
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    data_bytes = arr.tobytes()

    # Create a Reed–Solomon codec with the specified number of parity symbols.
    rsc = RSCodec(nsym)

    # Encode the data; this returns a bytes object which consists of the original data
    # followed by parity bytes.
    encoded: bytes = rsc.encode(data_bytes)

    # Split the encoded data evenly into n_fragments.
    fragment_size = len(encoded) // n_fragments
    fragments: List[bytes] = [
        encoded[i * fragment_size : (i + 1) * fragment_size] for i in range(n_fragments)
    ]

    # If there are leftover bytes, append them to the last fragment.
    if len(encoded) % n_fragments:
        fragments[-1] += encoded[n_fragments * fragment_size:]

    return fragments

def decode_shard(
    fragments: List[bytes],
    nsym: int,
    original_shape: Tuple[int, ...],
    original_dtype: Any = np.float32
) -> torch.Tensor:
    """
    Decode a set of Reed–Solomon encoded fragments to recover the original tensor shard.

    Args:
        fragments (List[bytes]): List of fragments (as bytes) that together form the encoded data.
        nsym (int): The number of parity symbols that were added during encoding.
        original_shape (Tuple[int, ...]): The original shape of the tensor.
        original_dtype: The data type of the original tensor (default is np.float32).

    Returns:
        torch.Tensor: The reconstructed tensor with the given shape and type.
    """
    # Concatenate all fragments to reconstruct the full encoded data.
    encoded: bytes = b"".join(fragments)

    # Create the RS codec and decode the data. The decode method returns a tuple;
    # the first element is the decoded data as bytes.
    rsc = RSCodec(nsym)
    decoded_bytes: bytes = rsc.decode(encoded)[0]

    # Convert the decoded bytes back into a NumPy array and reshape it.
    arr = np.frombuffer(decoded_bytes, dtype=original_dtype).copy()
    arr = arr.reshape(original_shape)
    return torch.from_numpy(arr)

def main() -> None:
    shard = torch.randn(100, 100, dtype=torch.float32)
    fragments = encode_shard(shard, n_fragments=10, nsym=32)
    print("Encoded fragment lengths:", [len(frag) for frag in fragments])
    recovered_shard = decode_shard(fragments, nsym=32, original_shape=shard.shape)
    print("Recovery successful:", torch.allclose(shard, recovered_shard, atol=1e-6))
