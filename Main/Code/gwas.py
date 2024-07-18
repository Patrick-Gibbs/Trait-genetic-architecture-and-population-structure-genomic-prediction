"""Runs GWAS for every trait recorded"""

import os
from Main.HelperClasses.GetAraData import *
getAraData = GetAraData(path_to_data='./data')
PATH_TO_VCF = getAraData.get_vcf_path()
GENE_FILE =  getAraData.go_terms_path()
SAVE_DIR = "Main/results/gwas/"

traits = list(pd.read_csv('Main/results/detailed_scores.csv')['trait'])
traits += [
    'study_126_Trichome_stem_length'
]

for trait in traits:
    getAraData.save_phenotype_file_for_GWAS(trait)
    pheno_file = getAraData.get_gwas_path(trait)
    output = f'{SAVE_DIR}{trait}'
    os.system("vcf2gwas -v %s -pf %s --minaf 0.05 -gf %s -o %s -ap -lmm" % (PATH_TO_VCF, pheno_file, GENE_FILE, output))