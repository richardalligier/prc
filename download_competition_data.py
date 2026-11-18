import argparse
import json
from subprocess import call
import os

def main():
    parser = argparse.ArgumentParser(
                    description='util to download from repo',
    )
    parser.add_argument("-filename")
    parser.add_argument("-out")
    parser.add_argument("-json")
    args = parser.parse_args()
    with open(args.json) as f:
        d = {x['name']:x['contentUrl'] for x in json.load(f)["@graph"] if x['@type']=='File'}
    cmd = f'wget --no-use-server-timestamps {d[args.filename]} -O {args.out}'
    #print(cmd)
    call(cmd,shell=True)
    #raise Exception

if __name__ == '__main__':
    main()    