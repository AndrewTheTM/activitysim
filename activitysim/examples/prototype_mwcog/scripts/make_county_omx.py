import pandas as pd
import argparse
import openmatrix as omx
import os
import numpy as np

def read_land_use(land_use_file, index_field_name, field_name, output_file):
    if not os.path.isfile(land_use_file):
        raise RuntimeError("Land use file not found")
    if land_use_file[-4:].lower() != ".csv":
        raise RuntimeError("Land use file needs to be a CSV")
    land_use = pd.read_csv(land_use_file)
    if not field_name in land_use.columns:
        raise RuntimeError("Field name is not in file")
    if not index_field_name in land_use.columns:
        raise RuntimeError("Index field name is not in file")
    co_output = np.tile(land_use[field_name], reps = (land_use.shape[0], 1))
    mapping = np.array(land_use[index_field_name])
    m = omx.open_file(output_file, 'w')
    m.create_mapping('taz', mapping)
    m['COUNTY'] = np.array(co_output)
    m.close()
    print("Process complete!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='preprocessor')
    parser.add_argument(
        '-l', '--land_use_file',
        action='store', help='Land use file',
        required = True)
    parser.add_argument(
        '-f' , '--field_name',
        action = 'store', 
        help='Land use county field',
        required = True)
    parser.add_argument(
        '-i' , '--index_field_name',
        action = 'store', 
        help='Index field',
        required = True)
    parser.add_argument(
        '-o', '--output_file',
        action = 'store',
        help = 'Output file name',
        required = True
    )
    args = parser.parse_args()
    read_land_use(args.land_use_file, args.index_field_name, args.field_name, args.output_file)
