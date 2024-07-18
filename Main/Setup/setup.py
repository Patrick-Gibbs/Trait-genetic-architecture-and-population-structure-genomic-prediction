import  os
import pandas as pd
import statsmodels.api as sm
import sys

""" 
This gets all of the phenotypes from the arapheno database, then parses them into
a phenotypes directory.
"""

# first we downloaded all the phenotypes from the arapheno database (as of march 2023)
# if rerunning this code we the version of the database in this repo, as new traits are
# continuely being added
path_to_data = './data'
path_to_database = f"{path_to_data}/arapheno_database"
os.system.mkdir(path_to_data)

def get_phenotypes(): 
    # to redownload the database uncomment the following lines
    os.system(f"mkdir {path_to_database}")
    os.system(f"wget https://arapheno.1001genomes.org/static/database.zip -P {path_to_database}")
    os.system(f"unzip {path_to_database}/database.zip -d {path_to_database}")

    # next we exactact the phenotypes from the database and put them into a directory
    path_to_downloaded_traits = f"{path_to_data}/arapheno_downloaded_phenotypes"
    os.system(f"mkdir {path_to_downloaded_traits}")

    studies = sorted([e for e in os.listdir(path_to_database) if e.isnumeric()], key=lambda x: int(x))
    for study in studies:
        study_results = pd.read_csv(f"{path_to_database}/{study}/study_{study}_values.csv")
        # filter to with expression
        phenotype_names = study_results.columns[2:]
        for phenotype in phenotype_names:
            indicies = study_results[phenotype].isna()
            df = pd.DataFrame({'accession_id': study_results['accession_id'], 'phenotype_value': study_results[phenotype]})[~indicies].sort_values(by='accession_id')
            # remove non-alphanumeric characters from phenotype name
            phenotype = ''.join([letter for letter in phenotype.replace(' ', '_') if 
                                    (letter.isalnum() or letter in '_-') and letter.isascii()])
            df.to_csv(f"{path_to_downloaded_traits}/study_{study}_{phenotype}.csv", index=False)

    """
    note 'ara_herbavore_resistance_phenotypes' contains 22 metabolic traits, which are the concentration of glucosinolates 
    involved in the Arabidopsis thaliana herbivory stress response. This data is not available online but was
    provided to us by the authors of the following paper:
    Brachi, B., Meyer, C. G., Villoutreix, R., Platt, A., Morton, T. C., Roux, F., & Bergelson, J. (2015). 
    Coselected genes determine adaptive variation in herbivore resistance throughout the native range of Arabidopsis thaliana. 
    Proceedings of the National Academy of Sciences, 112(13), 4032-4037. https://doi.org/10.1073/pnas.1421416112
    """


    """
    The next section of code reads the phenotypes from the arapheno_downloaded_phenotypes directory and formats them further:
        -drops duplicates rows. 
        -where there are multiple acessions corosponding to the same accession_id (i.e. multiple phenotypes for the same accession) 
        phenotype_value is set the genetic effect derived from a linear mixed model
    """


    outgoing_path = "data/arapheno_downloaded_phenotypes_cleaned/"
    os.system(f"mkdir {outgoing_path}")
    for incoming_path in [path_to_downloaded_traits + '/', 'data/ara_herbavore_resistance_phenotypes/']:
        # get csv file names
        csv_files = os.listdir(incoming_path)
        # filter csv_files to only csvs
        csv_files = [file for file in csv_files if file.endswith(".csv")]
        for csv_file in csv_files:
            # read csv
            current_csv = pd.read_csv(incoming_path + csv_file)
            # drop duplicates
            current_csv = current_csv.drop_duplicates()
            # if there are multiple phenotypes for the same accession, set phenotype_value to the genetic effect
            if len(set(current_csv['accession_id'])) != len(current_csv):
                # find genetic effect using mixed linear model
                phenotype_value = current_csv['phenotype_value']
                formula = "phenotype_value ~ 1 + (1 | accession_id)"
                model = sm.MixedLM.from_formula(formula, data=current_csv, groups=current_csv["accession_id"])
                results = model.fit()
                # get the genetic effects from the model
                genetic_effect = results.random_effects
                # turn genetic effects into a dataframe of two columns
                items = [(g, list(y)[0]) for g, y in genetic_effect.items()]
                # create a dataframe from the random effects
                group_effects_dict = {"accession_id": [x[0] for x in items], "phenotype_value": [x[1] for x in items]}
                reformatted_df = pd.DataFrame(group_effects_dict)
            else:
                reformatted_df = current_csv
            # write reformatted_df to csv
            reformatted_df.to_csv(outgoing_path + csv_file)
            # write trait also in plink format for downstream gwas

def make_temp_dirs():
    # make the following dirs:
    dirs = ['ara_filtered_genome_indexs',
        'arapheno_database' , 
        'bimbam', 
        'ld_filtered_genomes_temp_files'
        'ara_gwas_pheno', 
        'arapheno_downloaded_phenotypes',
        'functional_data', 
        'trait_accessions'
        'ara_herbavore_resistance_phenotypes', 
        'arapheno_downloaded_phenotypes_cleaned', 
        'ld_filtered_genomes'
    ]
    for dir in dirs:
        os.system(f"mkdir {path_to_data}/{dir}")

def get_genotypes():
    """
    The next section downloads the bim/bam files
    """
    os.system(f"wget https://figshare.com/ndownloader/files/20135279 -P {path_to_data}/bimbam/k2029.bed")
    os.system(f"wget https://figshare.com/ndownloader/files/20135282 -P {path_to_data}/bimbam/k2029.bim")
    os.system(f"wget https://figshare.com/ndownloader/files/20135285 -P {path_to_data}/bimbam/k2029.fam")
    os.system(f"wget https://figshare.com/ndownloader/files/20135384 -P {path_to_data}/bimbam/k2029.vcf.gz")

# get phenotypes is commented out becouse the resultant data is already in the git
# get_phenotypes()

# get_genotypes is not commented becouse the bim/bam files are too big to go on the git and thus must
# be obtained via wget
make_temp_dirs()
get_genotypes()