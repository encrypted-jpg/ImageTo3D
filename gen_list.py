import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filelist', type=str, default='finalfilelist.txt')
parser.add_argument('--top', type=int, default=8)
parser.add_argument('--count', type=int, default=1250)
parser.add_argument('--train', type=float, default=0.8)
parser.add_argument('--test', type=float, default=0.1)
parser.add_argument('--val', type=float, default=0.1)
parser.add_argument('--dir', type=str, default='.')


def get_args():
    return parser.parse_args()


def load_file(filelist):
    with open(filelist, 'r') as f:
        lines = f.read().split("\n")
    return lines


def get_json(filelist, top, count):
    files = load_file(filelist)
    cats = {x.split("-")[0]: [] for x in files}
    for x in files:
        cats[x.split("-")[0]].append("-".join(x.split("-")[1:]))

    ckeys = sorted(cats, key=lambda k: len(cats[k]), reverse=True)
    ckeys.remove("")

    jsondict = {}
    for key in ckeys[:top]:
        jsondict[key] = cats[key][:count]

    return jsondict


def get_split(filelist, top, total_count, train_ratio, test_ratio, val_ratio):
    jsondict = get_json(filelist, top, total_count)

    final = {}

    train_ct = 0
    test_ct = 0
    val_ct = 0

    for key in jsondict:
        final[key] = {"train": [], "test": [], "val": []}
        final[key]["train"] = jsondict[key][:int(
            len(jsondict[key]) * train_ratio)]
        final[key]["test"] = jsondict[key][int(
            len(jsondict[key]) * train_ratio):int(len(jsondict[key]) * (train_ratio + test_ratio))]
        final[key]["val"] = jsondict[key][int(
            len(jsondict[key]) * (train_ratio + test_ratio)):]

        print(
            f"Key: {key}, Train: {len(final[key]['train'])}, Test: {len(final[key]['test'])}, Val: {len(final[key]['val'])}")

        train_ct += len(final[key]["train"])
        test_ct += len(final[key]["test"])
        val_ct += len(final[key]["val"])

    print(
        f"Top: {len(final.keys())}, Train: {train_ct}, Test: {test_ct}, Val: {val_ct}, Total: {train_ct + test_ct + val_ct}")

    return final


def save_jsons(final, dir):
    os.makedirs(dir, exist_ok=True)
    print(f"Saving to {dir}")
    with open(os.path.join(dir, "final.json"), 'w') as f:
        json.dump(final, f)

    # with open(os.path.join(dir, "train.json"), 'w') as f:
    #     json.dump(train, f)
    # with open(os.path.join(dir, "test.json"), 'w') as f:
    #     json.dump(test, f)
    # with open(os.path.join(dir, "val.json"), 'w') as f:
    #     json.dump(val, f)


if __name__ == "__main__":
    args = get_args()
    if args.train > 1 or args.test > 1 or args.val > 1 or args.train + args.test + args.val > 1 or args.train + args.test + args.val < 1:
        raise ValueError(
            "Train, test, and val ratios must be between 0 and 1 and must add up to 1")
    final = get_split(
        args.filelist, args.top, args.count, args.train, args.test, args.val)
    save_jsons(final, args.dir)
