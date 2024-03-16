import argparse
import datetime
import pathlib

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

def main():
    parser = argparse.ArgumentParser(
        description='Helper script for executing jupyter notebooks contained in this project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'notebook',
        type=str,
        help='Path to the notebook to execute.'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Number of times to execute the notebook.'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='python3',
        help='Name of the kernel to use for notebook execution.'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout in seconds for notebook cell execution.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./executed_notebooks',
        help='Directory to save the executed notebooks.'
    )
    args = parser.parse_args()

    notebook_path = pathlib.Path(args.notebook)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name=args.kernel)

    print(f'Executing notebook "{notebook_path}"...')
    t_start = datetime.datetime.now()

    for i in range(args.repeat):
        print(f'\nRun {i+1}/{args.repeat}...')
        loop_t_start = datetime.datetime.now()

        notebook_name = notebook_path.stem
        save_path = save_dir / f'(run {i+1}) {notebook_name}.ipynb'
        try:
            ep.preprocess(nb, {'metadata': {'path': './'}})
        except CellExecutionError as err:
            print(f'\tError occurred during execution: {err}.')
            print(f'\tSee notebook file "{save_path}" for the traceback.')
            prompt = input('\tContinue? (y/n): ')
            if prompt != 'y':
                print('\n\nAborted.')
                return
        finally:
            with open(save_path, mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        
        loop_t_end = datetime.datetime.now()
        print(f'\tDone. Time taken: {loop_t_end - loop_t_start}.')
    
    t_end = datetime.datetime.now()

    print(f'\n\nAll done. Total time elapsed: {t_end - t_start}.')
    print(f'Average time per run: {(t_end - t_start) / args.repeat}.')

if __name__ == '__main__':
    main()
