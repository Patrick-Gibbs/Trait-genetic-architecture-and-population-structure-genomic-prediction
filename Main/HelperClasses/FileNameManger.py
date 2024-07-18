import pandas as pd
class FileNameManager:
    def __init__(self, path_to_data='./data') -> None:
        # path to directory where generates represenations should be stored
        self.path_to_phenotypes_directory = path_to_data + '/arapheno_downloaded_phenotypes_cleaned/'
        self.directory_to_raw_phenotypes = path_to_data + '/arapheno_downloaded_phenotypes/'
        self.directory_to_save_genotypes = path_to_data + '/ara_genotypes/'
        self.directory_to_save_pca_genotypes = path_to_data + '/ara_PCA_genotypes/'
        self.path_to_phenotype_csv_for_gwas = path_to_data + '/ara_gwas_pheno/'
        self.path_to_2029_accessions = path_to_data + "/bimbam/accessions_2029"
        self.ara_corrosponding_accessions = path_to_data + "/ara_corrosponding_accessions"
        self.path_to_corrosponding_filterd_indexs = path_to_data + '/ara_filtered_genome_indexs'
        self.genome_map_file_name = path_to_data + '/bimbam/k2029.map'
        self.path_to_ld_filtered_genomes = path_to_data +  '/ld_filtered_genomes'
        self.path_to_ld_filtered_genomes_temp_files = path_to_data + '/ld_filtered_genomes_temp_files'
        self.path_to_bim_bam_files = path_to_data + '/bimbam/k2029'
        self.path_to_vcf_file = path_to_data + '/bimbam/k2029.vcf.gz'
        self.trait_accessions_path = path_to_data + '/trait_accessions/'
        self.path_to_functional_data = path_to_data + '/functional_data/'
