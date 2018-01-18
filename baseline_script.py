import io, sys, os, csv
import xml.etree.ElementTree as et
import logging
from dicttoxml import dicttoxml
import pandas

input_dir = str(sys.argv[1])
output_dir = str(sys.argv[2])

# 3 datasets different algorithms here
# for now, let's output baseline results 

# here's a list of all our IDs stripped of file extension
profile_csv = "{}/{}".format(input_dir, "profile/profile.csv")
profile_df = pandas.read_csv(profile_csv)

# XML is hella frustrating, why not JSON or a single csv??
for i in profile_df.userid:
    elem = et.Element("user", attrib={
        "id": i,
        "age_group": "xx-24",
        "gender": "male",
        "extrovert": "4.0",
        "neurotic": "3.0",
        "agreeable": "4.0",
        "conscientious": "3.5",
        "open": "4.0"})
    tree = et.ElementTree(element=elem)
    filename = "{}/{}.xml".format(output_dir, i)
    with open(filename, 'wb') as f:
        tree.write(f)


