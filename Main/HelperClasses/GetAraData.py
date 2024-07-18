import functools
import operator
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os
from bed_reader import open_bed
from sklearn.decomposition import PCA
from Main.HelperClasses.FileNameManger import FileNameManager
from sklearn.model_selection import PredefinedSplit

"""
This script defined GetAraData Object, which is used as a getter for all the data in the project.
"""

PATH_TO_PLINK = 'plink'

class GetAraData(FileNameManager):
    def __init__(self, path_to_data='./data', maf=0.03, r2=0.8, window_kb=10):
        """
        A getter object that can retrieve any arabidopsis data from the data folder.
        `path_to_data`: path to the data folder
        `maf`: minor allele frequency to filter by
        `r2`: r2 to filter by
        """
        super().__init__(path_to_data)
        self.maf = maf
        self.r2 = r2
        self.window_kb = window_kb

        # default sizes to generate for genome partitioning
        self.SET_OF_PARTITION_SIZES = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 23, 25, 28, 30, 32, 35, 40, 50, 60, 75, 90,
                                       100, 140, 250, 300, 500, 750, 1000]


    def get_vcf_path(self):
        """returns the path to the vcf file"""
        return self.path_to_vcf_file


    def get_file_suffix(self, trait: str) -> str:
        """
        returns a suffix to be added to the end of a file name to indicate the trait
        but does not add anything for dummy genomes
        """
        if trait == 'dummy':
            return ''
        else:
            return f'_ld_filtered_maf{self.maf}_windowkb{self.window_kb}_r2{self.r2}'
    
    def get_filtered_traits(self):
        return np.load(self.path_to_ontology_results + '/filtered_traits.npy', allow_pickle=True)

    def get_best_trait_model(self):
        return pd.read_csv(self.path_to_ontology_results + '/best_models.csv')

    def trait_name_to_corrosponding_filtered_indexs_path(self, trait_name: str, mask = None) -> str:
        """takes a trait name and turns it into a file name to store corrosponding indexs"""
        path = f"{self.path_to_corrosponding_filterd_indexs}/{trait_name}_corrosponding_filtered_indexs{self.get_file_suffix(trait_name)}"
        if mask is None:
            return path + ".npy"
        else:
            return path + str(mask) + ".npy"

    def go_terms_path(self) -> str:
        """returns the path to the go terms"""
        return self.path_to_functional_data + 'ATH_GO_GOSLIM.txt'

    def trait_name_to_csv_file_name(self, trait_name: str) -> str:
        """takes a trait name and turns it indo a csv file name"""
        return trait_name + ".csv"

    def trait_name_to_numpy_name(self, trait_name: str) -> str:
        """takes a trait name, and turns it ento a file name that to store phenotype as np array"""
        return trait_name + 'phenotype_values.npy'

    def phenotype_file_path(self, trait_name: str) -> str:
        """takes a trait name and turns it into a file name to store phenotype"""
        return self.directory_to_raw_phenotypes + self.trait_name_to_numpy_name(trait_name)

    def get_ld_filtered_genome_path(self, trait, mask=None):
        if mask is None:
            return f"{self.path_to_ld_filtered_genomes}/{trait}{self.get_file_suffix(trait)}.npy"
        else:
            return self.get_path_to_mask_genome(mask, trait)

    def trait_name_to_PCA_genome_matrix(self, trait_name: str, variance: float) -> str:
        """takes a trait name and turns it into a file name to store PCA decompersition of SNP matrix"""
        return self.directory_to_save_pca_genotypes + trait_name + "_PCA" + str(variance) + self.get_file_suffix(trait_name) + ".npy"
                                       
    def save_phenotype_file_for_GWAS(self, trait):
            """takes a trait name and saves a csv file in the format required for GWAS"""
            df = pd.read_csv(self.path_to_phenotypes_directory +
                            self.trait_name_to_csv_file_name(trait))
            new_csv = pd.DataFrame({'': df['accession_id'], trait: df['phenotype_value']})
            new_csv.to_csv(self.path_to_phenotype_csv_for_gwas + trait, index=False)

    def get_gwas_path(self, trait):
        """returns the path to the gwas file"""
        return self.path_to_phenotype_csv_for_gwas + trait

    def save_dummy_genome(self):
        """saves a simulated genome"""

        # dummy genome and phenotype are used for testing
        dummy_genome_file_name = self.get_ld_filtered_genome_path("dummy")
        dummy_phenotype_file_name = self.directory_to_raw_phenotypes + self.trait_name_to_numpy_name("dummy")

        # generate the dummy genome and phenotype
        X, y = self.generate_dummy_genome()
        with open(dummy_genome_file_name, 'wb') as f:
            np.save(f, X)
        with open(dummy_phenotype_file_name, 'wb') as f:
            np.save(f, y)

    def get_raw_file_name(self, trait):
        return self.path_to_phenotypes_directory + "/" + trait + ".csv"

    def get_trait_names(self):
        """returns a list of trait names"""
        files = os.listdir(self.path_to_phenotypes_directory)
        names = [file.replace(".csv", "") for file in files if '.csv' in file]
        return names

    def get_heritable_traits(self):
        """returns a list of heritable traits"""
        return self.heritable_traits

    def get_phenotype(self, name: str):
        """returns a numpy array of phenotype values for trait `name`"""
        if name == 'dummy':
            return self.get_dummy_phenotype()

        data_frame = pd.read_csv(self.get_raw_file_name(name))
        pheotyped_accessions = self.get_k2029_accessions_with_phenotype(name)
        data_frame = data_frame.sort_values(by=['accession_id'])
        data_frame = data_frame[data_frame['accession_id'].isin(pheotyped_accessions)]

        y = np.array(data_frame['phenotype_value'])
        # old
        #file_path = self.phenotype_file_path(name)
        #y = self.load_or_generate_features(file_path, lambda: self.save_genomes_k2029(traits=[name]))

        return y
    
    def get_normalised_phenotype(self, name:str):
        """returns a numpy array of the phenotype normalised"""
        y = self.get_phenotype(name)
        return (y - np.mean(y))/np.std(y)

    def get_trait_accessions(self, trait):
        """returns the accession ids for the specified trait"""
        data_frame = pd.read_csv(self.get_raw_file_name(trait))
        accesions = np.array(list(data_frame['accession_id']))
        return accesions

    def get_k2029_accessions(self):
        """gets all accession numbers from the k2029 expermiment"""
        accessions = np.loadtxt(self.path_to_2029_accessions, dtype=int)
        return accessions

    def get_k2029_accessions_with_phenotype(self, trait):
        """gets all the acessions from the k2029 dataset that has an assioated trait with it"""
        accessions_k2029 = np.loadtxt(self.path_to_2029_accessions)
        accessions_trait = self.get_trait_accessions(trait)
        accessions = np.intersect1d(accessions_k2029, accessions_trait)
        return accessions.astype(int)

    def my_PCA(self, X: np.ndarray, variance_maintained) -> np.ndarray:
        """
            returns the PCA reduced matrix of `X` with `variance_maintained` variance maintained
            if `variance_maintained` is 1.0, then the number of components is the number of features
        """
        if variance_maintained == 1.0 or variance_maintained == 1:
            variance_maintained = min(X.shape[1], X.shape[0])
        pca = PCA(n_components=variance_maintained)
        pca.fit(X)
        X_reduced = pca.transform(X)
        return X_reduced

    def save_pca_feature_reduction_on_SNPs(self, traits="All", variance_maintained=0.95, use_python=False):
        """for each `trait` in `traits` saves a file containing the PCA reduced SNPs.
        `variance_maintained` is the amount of variance to be maintained in the PCA reduction.
        uses julia becouse python is bugging for this transformation.
        """
        if traits == "All":
            traits = self.get_trait_names()
        for trait in traits:
            output_path = self.trait_name_to_PCA_genome_matrix(trait, variance=variance_maintained)
            
            X = self.get_genotype(trait)
            X_pc = self.my_PCA(X, variance_maintained)
            f = open(output_path, 'wb')
            np.save(f, X_pc)
            f.close()

    def load_or_generate_features(self, file_path, data_generator, read_mode='rb', write_mode='wb'):
        """loads and returns a dataset with `file_path`, if the file is not found
        then it is generated using `data_generator` function"""
        try:
            f = open(file_path, read_mode)
        except FileNotFoundError:
            print(file_path, "not found, generating data using", data_generator)
            data_generator()
            f = open(file_path, read_mode)
        X = np.load(f)
        f.close()
        return X

    def get_pca_feature_reduced_SNPs(self, trait, variance_maintained=0.95):
        X = self.load_or_generate_features(self.trait_name_to_PCA_genome_matrix(trait, variance=variance_maintained),
                                           lambda: self.save_pca_feature_reduction_on_SNPs([trait],
                                                                                           variance_maintained=variance_maintained))
        return X

    def generate_dummy_genome(self, genome_size=100, num_individuals=100, QTL_number=10, noise=0.001, QTL_effect_size=1):
        """generates a simulated genome"""
        X = np.random.randint(2, size=(num_individuals, genome_size))
        # QTLs are normally distributed, with mean 0 and variance 1, corrosponding to random spots in the genome.
        # QTLs are the features that determine the phenotype
        QTLs = np.zeros(genome_size)
        # create  `QTL_number` QTLs, the effect size is normally distributed with mean 0 and variance 1
        QTLs[np.random.randint(0, genome_size, QTL_number)] = np.random.normal(0, QTL_effect_size, QTL_number)

        # the phenotype is the sum of the QTLs
        y = X @ QTLs

        # add some noise to the phenotype
        y += np.random.normal(0, noise, num_individuals)

        return X, y

    def get_dummy_genome(self):
        """get the dummy genome"""
        X = self.load_or_generate_features(self.get_ld_filtered_genome_path('dummy'), 
            lambda: self.save_dummy_genome())
        return X

    def get_dummy_phenotype(self):
        """get the dummy phenotype"""
        y = self.load_or_generate_features(self.directory_to_raw_phenotypes + self.trait_name_to_numpy_name('dummy'), lambda: self.save_dummy_genome())
        return y

    def get_genotype(self, trait_name: str, mask=None):
        """get the SNPS, pertaining to indiduals that have 
        had their pheotypes measured for a specifed trait"""

        if trait_name == "dummy":
            return self.get_dummy_genome()    

        return self.load_or_generate_features(self.get_ld_filtered_genome_path(trait_name, mask),
                                            lambda: self.save_ld_filtered_genome(trait_name, mask))

    def get_corrosponding_filterd_indexs(self, trait, mask=None):
        """get the indexs of the SNPs that are not filtered out"""
        pos = np.load(self.trait_name_to_corrosponding_filtered_indexs_path(trait, mask))
        return pos

    def get_genome_possitions_from_filtered_indexes(self, trait, mask=None):
        indexes = self.get_corrosponding_filterd_indexs(trait, mask=mask)
        genome_locations = self.get_genome_map()
        filtered_genome_locations = genome_locations[indexes]
        return filtered_genome_locations

    def get_GFF(self):
        """get the GFF file which stores the annoation infomation for Arabidposis"""
        return pd.read_csv(self.GFF_file_nam, sep='\t', low_memory=False)

    def get_pca_features(
        self,
        trait,
        pc_variences_to_test = [0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99],
        test_pca_snps = True,
        get_ibs = True,
        get_ldak = True
    ):
        """get the PCA reduced SNPs and kinships
        input:
            trait: the trait to get the PCA reduced SNPs and kinships for
            pc_variences_to_test: the variances to test for the PCA
            test_pca_snps: if the PCA should be tested on the SNPs
        output:
            feature_representations: the PCA reduced SNPs and kinships
            varience_mained_in_PCA: the variances that were tested for the PCA
            name_of_feature_representations: the names of the PCA reduced SNPs and kinships
        """
        varience_mained_in_PCA = []
        feature_representations = []
        name_of_feature_representations = []
        for num_pcs in pc_variences_to_test:
            # gets PCA reduced SNPs
            if test_pca_snps:
                snps_pca = self.get_pca_feature_reduced_SNPs(trait, variance_maintained=num_pcs)
                feature_representations.append(snps_pca)
                name_of_feature_representations.append("PCA_SNPs")
            if get_ibs:
                kinships_pca = self.get_genome_simularity(trait, PCA_variance=num_pcs, method=KinshipType.IBS)
                feature_representations.append(kinships_pca)
                name_of_feature_representations.append("PCA_Kinship_IBS")

            if get_ldak:
                kinships_pca = self.get_pca_kinship_matrix(trait, variance_maintained=num_pcs, method=KinshipType.LDK)
                feature_representations.append(kinships_pca)
                name_of_feature_representations.append("PCA_Kinship_LDK")

            # storing PCs used
            varience_mained_in_PCA += [num_pcs]*sum([get_ibs, get_ldak, test_pca_snps])
    
        return feature_representations, varience_mained_in_PCA, name_of_feature_representations

    def save_ld_filtered_genomes(self, traits):
        for trait in traits:
            self.save_ld_filtered_genome(trait, maf=self.maf, window_kb=self.window_kb, r2=self.r2)

    def get_prune_in_file_path(self, trait, mask=None):
        path = f"{self.path_to_ld_filtered_genomes_temp_files}/{trait}_ld_filtered_maf_{self.maf}_window_kb_{self.window_kb}_r2_{self.r2}"
        if mask is not None:
            path = path + str(mask)

        path = path.replace(' ', '_')
        path = path.replace(',', '_')
        path = path.replace('[', '')
        path = path.replace(']', '')
        path = path.replace("'", '')
        path = path.replace('"', '')
        return path
            
    def get_prune_in_file(self, trait, mask=None, repeat=True):
        try:
            return pd.read_csv(f"{self.get_prune_in_file_path(trait, mask=mask)}.prune.in", sep='\t', names=['id'])
        except FileNotFoundError:
            if repeat:
                self.save_ld_filtered_genome(trait, mask=mask)
                return pd.read_csv(f"{self.get_prune_in_file_path(trait, mask=mask)}.prune.in", sep='\t', names=['id'])
            else:
                raise FileNotFoundError


    def get_trait_accessions_path(self, trait):
        return f"{self.trait_accessions_path}{trait}" 

    def save_ld_filtered_genome(self, trait, mask=None):
        # saves a accession fam file to be read by plink
        #return map_file
        
        maf, window_kb, r2 = self.maf, self.window_kb, self.r2

        
        acc = self.get_trait_accessions(trait)
        acc = np.array([acc,acc]).T.astype(int)
        np.savetxt(self.get_trait_accessions_path(trait), acc, fmt='%i')
        bimbam = self.path_to_bim_bam_files
        prune_file_path = self.get_prune_in_file_path(trait, mask)
        prune_file_path = prune_file_path.replace('(', '\(')
        prune_file_path = prune_file_path.replace(')', '\)')
        print(prune_file_path)

        print('step',1)
        if mask is None:
            print(f"{PATH_TO_PLINK} --bfile {bimbam} --keep {self.get_trait_accessions_path(trait)} --maf {maf} --indep-pairwise {window_kb} kb 1 {r2} --out {prune_file_path}")
            if r2 == 1:
                r2=0.9999
            os.system(f"{PATH_TO_PLINK} --bfile {bimbam} --keep {self.get_trait_accessions_path(trait)} --maf {maf} --indep-pairwise {window_kb} kb 1 {r2} --out {prune_file_path}")

        else:
            print('step',2)
            include_file_path = mask.get_path_to_include()
            print(include_file_path)
            include_file_path = include_file_path.replace('(', '\(')
            include_file_path = include_file_path.replace(')', '\)')
            print(include_file_path)

            print(f"{PATH_TO_PLINK} --bfile {bimbam} --keep {self.get_trait_accessions_path(trait)} --maf {maf} --indep-pairwise {window_kb} kb 1 {r2} --extract {include_file_path} --out {prune_file_path}")
            os.system(f"{PATH_TO_PLINK} --bfile {bimbam} --keep {self.get_trait_accessions_path(trait)} --maf {maf} --indep-pairwise {window_kb} kb 1 {r2} --extract {include_file_path} --out {prune_file_path}")
        prune_in = self.get_prune_in_file(trait, mask, repeat=False)
        print('step',3)

        map_file = pd.read_csv(f"{bimbam}.map", sep='\t', names = ['chr', 'id', 'index'])
        all_ids = map_file['id']
        ids_of_pruned = prune_in['id']
        indexes_to_keep = []
        
        # finding prune indexes
        i,j = 0,0
        print("finding prune indexes")
        while i < len(all_ids) and j < len(ids_of_pruned):
            if i % 1000000 == 0: print(i, end=' ')
            if  all_ids[i] == ids_of_pruned[j]:
                indexes_to_keep.append(i)
                j += 1
            i += 1

        phenotyped_accessions = self.get_k2029_accessions_with_phenotype(trait).astype(int)
        all_accessions = self.get_k2029_accessions().astype(int)
        rows_to_keep = []
        i,j = 0,0

        while i < len(all_accessions) and j < len(phenotyped_accessions):
            if all_accessions[i] == phenotyped_accessions[j]:
                rows_to_keep.append(i)
                j += 1
            i += 1
        rows_to_keep = np.array(rows_to_keep)

        indexes_to_keep = np.array(indexes_to_keep)

        print("saving indexes to kept at ", self.trait_name_to_corrosponding_filtered_indexs_path(trait, mask))
        np.save(self.trait_name_to_corrosponding_filtered_indexs_path(trait, mask), indexes_to_keep)

        max_index = np.max(indexes_to_keep)
        print("number of markers to keep", len(indexes_to_keep))

        print("reading bim/bam files")

        bed = open_bed(f"{bimbam}.bed")
        X = bed.read(index=(rows_to_keep, indexes_to_keep))
        
        print("clipping the genome between 0,1 remove this step if you are using a genome that is not in 0,1")
        X = np.clip(X,0,1)
        
        print("saving genome")
        path = self.get_ld_filtered_genome_path(trait, mask)
        np.save(path, X)
        print('done!')

    def get_chromosome(self, trait, mask=None):
        """Returns a list of the chromosomes in the genome"""
        pos = self.get_genome_possitions_from_filtered_indexes(trait, mask=mask)
        intersect = []
        for i in range(len(pos) - 1):
            if pos[i] > pos[i+1]:
                intersect.append(i + 1)
        return intersect

    def get_genome_map(self, return_full_map=False):
        """get the genome map file which stores the possition of each SNP on each chromosome"""
        full_map = pd.read_csv(self.genome_map_file_name, sep='\t')
        possitions = full_map['pos'].to_numpy()
        if return_full_map:
            return full_map
        return possitions