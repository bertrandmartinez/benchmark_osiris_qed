
def replace(string, path):

    replacer_file = open(path, "rt")

    replacer = replacer_file.read()

    fout = open(
        "/Users/bertrand/Documents/osiris/main/main_bertrand/source/qed/os-qed-coulomb.f03", "rt")

    file = fout.read()

    file = file.replace(string, replacer)

    fout.close()

    fout = open(
        "/Users/bertrand/Documents/osiris/main/main_bertrand/source/qed/os-qed-coulomb.f03", "wt")

    fout.write(file)

    fout.close()

#replace("replace_axis_elec",  "tables/axis_elec.txt")
replace("replace_axis_pair",  "tables/axis_pair.txt")
#replace("replace_axis_posi",  "tables/axis_posi.txt")
#replace("replace_cs_Z_all",  "tables/cs_Z_all.txt")
replace("replace_cdf1_Z_all",  "tables/cdf1_Z_all.txt")
#replace("replace_cdf2_Z_all",  "tables/cdf2_Z_all.txt")
