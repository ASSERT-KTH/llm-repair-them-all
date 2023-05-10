from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils.benchmarks import get_benchmark
from core.utils.jsonl import write_jsonl
from core.benchmarks.bug import Bug
from strategies.zero_shot_single_hunk import ZeroShotSingleHunkPrompting
from typing import Optional, Union

import fire
import sys
import tqdm
import logging

def generate_sample(bug: Bug, prompt_strategy: str) -> dict[str, Optional[Union[str, Bug]]]:
    """
    Generates the sample for the given bug with the given prompt strategy.
    """

    # TODO: We only support single-hunk bugs for now
    prompt_strategy_obj = ZeroShotSingleHunkPrompting()
    prompt = prompt_strategy_obj.prompt(bug)
    # Check if prompt was generated
    if prompt is None:
        return {
            "identifier": bug.get_identifier(),
            "buggy_code": None,
            "fixed_code": None,
            "prompt_strategy": prompt_strategy,
            "prompt": None,
            "ground_truth": bug.get_ground_truth()
        }
    
    # Unpack the prompt
    buggy_code, fixed_code, prompt = prompt
    return {
        "identifier": bug.get_identifier(),
        "buggy_code": buggy_code,
        "fixed_code": fixed_code,
        "prompt_strategy": prompt_strategy,
        "prompt": prompt,
        "ground_truth": bug.get_ground_truth()
    }


def entry_point(
    benchmark: str,
    prompt_strategy: str,
    n_workers: int = 4
):
    """
    Generates the test samples for the bugs of the given benchmark with the given
    prompt strategy, and writes the results to f"samples_{dataset}_{prompt_strategy}.jsonl.gz"
    """

    # Get the benchmark, check if it exists, and initialize it
    benchmark_obj = get_benchmark(benchmark)
    if benchmark_obj is None:
        raise ValueError(f"Unknown benchmark {benchmark}")
    benchmark_obj.initialize()

    # Generate the prompts in parallel
    logging.info("Building the prompts...")
    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        # Launch a thread for each bug
        for bug in benchmark_obj.get_bugs():
            args = (bug, prompt_strategy)
            futures.append(executor.submit(generate_sample, *args))
            # TODO: debug
            # break

        # Check that all bugs are being processed
        #assert len(futures) == len(benchmark_obj.get_bugs()), "Some bugs are not being processed"

        # Wait for the results
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # Write results to jsonl file
    write_jsonl(f"samples_{benchmark}_{prompt_strategy}.jsonl.gz", results)

def main():
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(entry_point)


sys.exit(main())