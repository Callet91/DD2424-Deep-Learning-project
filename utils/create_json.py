"""Create JSON file from text file."""
import json as js


def create_json(filename):
    """Takes a .txt and and converts the the first two rows to a.

    .json
    """
    json_name = filename.split("/")
    json_name = json_name[-1]

    dict_tmp = {}

    with open(filename) as text_file:

        for line in text_file:
            line_split = line.strip().split("\t")

            dict_tmp[line_split[0]] = line_split[1].strip()

    out_file = open(json_name + ".json", "w")
    js.dump(dict_tmp, out_file, indent=4, sort_keys=False)
    out_file.close()

    print("Done with conversion to json...")
