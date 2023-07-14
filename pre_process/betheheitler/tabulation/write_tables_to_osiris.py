
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

Z = 29
Z_str = "{:d}".format(Z)
#replace("replace_axis_phot",  "tables/axis_phot.txt")
#replace("replace_axis_posi",  "tables/axis_posi.txt")
replace("replace_cdf_table", "tables/Z="+Z_str+"/cdf_Z_"+Z_str+".txt")
