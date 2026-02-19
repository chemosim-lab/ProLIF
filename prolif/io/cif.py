"""
I/O-related helper functions --- :mod:`prolif.io.cif`
=====================================================
This module provides a lightweight parser for Crystallographic Information File (CIF)
format.

Yu-Yuan (Stuart) Yang, 2025
"""

import shlex
from pathlib import Path

import pandas as pd


# A user can provide a custom CIF file with the standard amino acid.
def _block_decompose(data_block: list) -> tuple:
    """
    Decomposes a CIF data block into decriptive information and tables.

    .. versionadded:: 2.1.0
    """
    descriptions: list[str] = []
    data_tables: list[list] = []
    data_table: list[str] | None = None

    for idx, block_line in enumerate(data_block):
        if block_line.startswith("#"):
            if data_table is not None:
                # save the current table
                data_tables.append(data_table)
            # reset the table
            data_table = None
        elif block_line.startswith("loop_"):
            # table format
            data_table = []
        elif data_table is not None:
            # add data to the current table
            data_table.append(block_line)
            if idx == len(data_block) - 1:  # last line of the block
                # save the final table
                data_tables.append(data_table)
        else:
            descriptions.append(block_line)

    return descriptions, data_tables


def cif_parser_lite(cif_string: str) -> dict:
    """
    Parses a CIF string and returns a dictionary of data blocks.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    cif_string : str
        The CIF string to parse.

    """
    # Split the CIF string into blocks based on 'data_' lines
    data_blocks = {}
    current_block = None
    all_lines = cif_string.strip().split("\n")
    for idx, line in enumerate(all_lines):
        if line.startswith("data_"):
            current_block = line.split("data_")[1]
            data_block: list[str] = []
        elif line.startswith("##") or idx == len(all_lines) - 1:
            # end of a data block
            data_blocks[current_block] = data_block
        else:
            data_block.append(line.strip())

    # create a dictionary to hold the parsed data
    cif_dict: dict = {}
    for block_name, data_block in data_blocks.items():
        descriptions, data_tables = _block_decompose(data_block)
        cif_dict[block_name] = {"name": block_name}

        # descriptive information
        for each in descriptions:
            content = shlex.split(each)
            info_name = content[0].split(".")
            info = content[1]
            if info_name[0] not in cif_dict[block_name]:
                cif_dict[block_name][info_name[0]] = {}
            cif_dict[block_name][info_name[0]][info_name[1]] = info

        # data tables
        for data_table in data_tables:
            header = []
            data: list[list[str]] = []
            table_name = data_table[0].split(".")[0]
            for each_line in data_table:
                if each_line.startswith("_"):
                    # header line
                    header.append(each_line.split(".")[1].strip())
                else:
                    # data line
                    # Use shlex.split to respect quoted strings
                    row = [
                        item.strip('"')
                        if item.startswith('"') and item.endswith('"')
                        else item
                        for item in shlex.split(each_line, posix=False)
                    ]
                    data.append(row)

            table = pd.DataFrame(data, columns=header)
            cif_dict[block_name][table_name] = table

    return cif_dict


def cif_template_reader(cif_filepath: Path | str) -> dict:
    """
    Reads a CIF file and returns a dictionary of data blocks.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    cif_filepath : str
        The path to the CIF file to read.

    Returns
    -------
    dict
        A dictionary containing the parsed data blocks.

    """
    cif_string = Path(cif_filepath).read_text()

    return cif_parser_lite(cif_string)
