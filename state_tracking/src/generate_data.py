"""Generates data for the group sequence prediction task."""

import os
import random
from functools import reduce
from itertools import product, permutations
from pathlib import Path
import math

import fire
import polars as pl
import pyrootutils
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def group_reduce(lhs: str | int, rhs: int, G) -> int:  # noqa: N803
    """Reduce a sequence of group elements to a single element."""
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])

    return G.elements.index(prod)


def generate_group(g: (str, int)) -> FiniteAlgebra:
    """Generate an group from a string identifier."""
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


def main(
    group: str,
    k: int | list[int] = 10,
    samples: int | None = None,
    data_dir: str | Path = PROJECT_ROOT / "data",
    seed: int = random.randint(0, 1_000_000),
    overwrite: bool = False,
):
    """Generate data for the group sequence prediction task."""
    data_path = data_dir / f"{group}={k}.csv"
    if data_path.exists() and not overwrite:
        print(
            f"Data already exists at {data_path}. Use `--overwrite` to regenerate file."
        )
        return

    random.seed(seed)
    print(f"Using seed {seed}")

    # if group == "S5_only_swaps_hard":
    #     group_list = [generate_group(("S", 5))]
    #     group_prod = reduce(lambda x, y: x * y, group_list)
        
    #     # Get indices of two element swaps and identity
    #     ident = tuple(range(1, 6))
    #     perms = list(permutations(ident))
    #     allowed_indices = [i for i, perm in enumerate(perms) if sum(x != y for x, y in zip(ident, perm)) <= 2]

    #     num_elements = len(allowed_indices)
    #     num_unique_sequences = num_elements**k
    if group == "S5_only_swaps" or group == "S5_only_swaps_hard":
        group_list = [generate_group(("S", 5))]
        group_prod = reduce(lambda x, y: x * y, group_list)
        
        # Get indices of two element swaps and identity
        ident = tuple(range(1, 6))
        perms = list(permutations(ident))
        allowed_indices = [i for i, perm in enumerate(perms) if sum(x != y for x, y in zip(ident, perm)) <= 2]

        num_elements = len(allowed_indices)
        num_unique_sequences = num_elements**k
    elif 'limit_to' in group:
        g = group[0]
        n = int(group[1])
        limit = int(group[-1])
        assert 1 < limit < n, "Choose a suitable limit"
        assert g == "S", "Only works with permutation groups"
        
        group_list = [generate_group((g, n))]
        group_prod = reduce(lambda x, y: x * y, group_list)
        
        # Get indices of two element swaps and identity
        ident = tuple(range(1, n+1))
        perms = list(permutations(ident))
        allowed_indices = [i for i, perm in enumerate(perms) if sum(x != y for x, y in zip(ident, perm)) <= limit]
        
        expected_num_elements = sum([math.comb(n, l) for l in range(2,limit+1)]) + 1

        num_elements = len(allowed_indices)
        num_unique_sequences = num_elements**k
        # assert num_elements == expected_num_elements, f"num_elements: true={num_elements}, expected={expected_num_elements}"

    else:
        if "tokens" in group:
            g = group.split("_")[0]
            group_ids = [(g[0], int(g[1:]))]
        else:
            group_ids = [(g[0], int(g[1:])) for g in group.split("_x_")]
        for g in group_ids:
            assert g[0] in ["S", "Z", "A"], f"Groups must be one of S, Z, or A but was {g[0]}"
        group_list = [generate_group(g) for g in group_ids]
    
        group_prod = reduce(lambda x, y: x * y, group_list)
        num_elements = len(group_prod.elements)
        num_unique_sequences = num_elements**k
        allowed_indices = range(num_elements)

    print(f"allowed indices = {allowed_indices}")
    print(f"num_elements = {num_elements}") 

    if samples is None:
        print(
            f"Generating all {num_elements} ^ {k} = " f"{num_elements ** k} sequences."
        )
        print("Output data will not be shuffled.")

        sequences = product(allowed_indices, repeat=k)

    else:
        if samples > num_unique_sequences:
            print(
                f"Warning: {samples} > {num_unique_sequences}. I will only"
                f" generate {num_unique_sequences} examples."
            )
            samples = num_unique_sequences
        print(f"Randomly sampling {samples} sequences.")
        sequences = set()
        while len(sequences) < samples:
            sequences.add(tuple(random.choices(allowed_indices, k=k)))
        sequences = list(sequences)

    examples = []
    for seq in sequences:        
        if group.endswith('hard'):
            inputs = []
            acc = 0
            for i, x in enumerate(seq):
                acc = group_reduce(lhs=acc, rhs=x, G=group_prod)
                if i % 4 == 3:
                    inputs.extend((acc, 0, 0, 0))
                    acc = 0
                elif i == len(seq)-1:
                    inputs.extend((acc, *[0 for _ in range(i % 4)]))
            # target_indices = [i for i in range(len(inputs)) if i%4==3]
            
            acc = 0
            outputs = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in inputs]
            
            # shift outputs to make the model learn better
            outputs = [*(0,0,0), *(outputs[:-3])]

        elif "tokens" in group:
            new_length = len(seq)
            n_tokens = int(group.split('_')[1])
            seq = seq[:(new_length//n_tokens)+1]
            acc = 0
            out = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in seq]
            inputs = []
            outputs = []
            for x, y in zip(seq, out):
                inputs.append(x)
                outputs.append(y)
                if len(inputs) >= new_length:
                    break
                for i in range(n_tokens-1):
                    s_token = None
                    if "s_token" in group:
                        s_token = num_elements
                        inputs.append(s_token)
                        if not "only_input" in group:
                            outputs.append(s_token)
                        else:
                            outputs.append(y)
                    else:
                        inputs.append(x)
                        outputs.append(y)
                    if len(inputs) >= new_length:
                        break
                if len(inputs) >= new_length:
                    break
            # shift supervision
            fill = [0 for _ in range(n_tokens-1)]
            outputs = [*fill, *outputs[:-(n_tokens-1)]]
        else:
            inputs = seq
            # target_indices = list(range(len(inputs)))
            
            acc = 0
            outputs = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in inputs]

                
        assert len(inputs) == len(outputs), f"{len(inputs)}, {len(outputs)}, {len(seq)}"
                    
                
            
        
        examples.append(
            {
                "seed": seed,
                "input": " ".join(map(str, inputs)),
                "target": " ".join(map(str, outputs)),
                # "target_indices": " ".join(map(str, target_indices)),
            }
        )
    if "tokens" in group:
        print(f"n_tokens per symbol: {n_tokens}, fill:{fill}, s_token:{s_token}")
    
    ex_df = pl.from_dicts(examples)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f"Writing data to `{data_path}`")
    ex_df.write_csv(data_path)


if __name__ == "__main__":
    fire.Fire(main)
